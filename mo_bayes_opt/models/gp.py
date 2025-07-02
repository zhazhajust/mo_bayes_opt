import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
from botorch.models.gpytorch import GPyTorchModel

# ---------------- NoiseNet ----------------
class NoiseNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        noise = torch.exp(self.fc2(x))  # 输出正值
        return noise.squeeze(-1)

# ---------------- GP Model ----------------
class GPModel(gpytorch.models.ExactGP, GPyTorchModel):
    _num_outputs = 1

    def __init__(self, train_x, train_y, likelihood, input_dim):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, test_x):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad():
            pred_dist = self.__call__(test_x)
        # 恢复训练模式
        self.train()  
        self.likelihood.train()
        return pred_dist

    def predict_mean_std(self, test_x):
        pred_dist = self.predict(test_x)
        return pred_dist.mean, pred_dist.variance.sqrt()

# ---------------- GP Trainer with Adaptive Noise ----------------
class GPTrainer:
    def __init__(self, train_x, train_y, lr=0.01, num_train_iters=500, use_adaptive_noise=False, noise=1e-4):
        self.train_x = train_x
        self.train_y = train_y
        self.lr = lr
        self.num_train_iters = num_train_iters
        self.input_dim = train_x.shape[1]
        self.use_adaptive_noise = use_adaptive_noise

        if self.use_adaptive_noise:
            # 使用 NoiseNet 预测每个点的 noise
            self.noise_net = NoiseNet(self.input_dim)
            self.optimizer = torch.optim.Adam([
                {'params': self.noise_net.parameters(), 'lr': self.lr},
            ])
            self.likelihood = None  # 暂时延后初始化
            self.model = None
        else:
            # 使用固定 noise
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.model = GPModel(train_x, train_y, self.likelihood, input_dim=self.input_dim)
            self.model.likelihood.noise = noise
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self):
        for i in range(self.num_train_iters):
            self.optimizer.zero_grad()

            if self.use_adaptive_noise:
                # 更新 GP 模型每轮都基于当前 noise_net 输出
                estimated_noise = self.noise_net(self.train_x).detach()
                self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=estimated_noise)
                self.model = GPModel(self.train_x, self.train_y, self.likelihood, input_dim=self.input_dim)

                self.model.train()
                self.likelihood.train()

                # 新 optimizer for GP model
                gp_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
                gp_optimizer.zero_grad()

                output = self.model(self.train_x)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
                loss = -mll(output, self.train_y)

                # 联合训练：GP loss + NoiseNet L2 正则项（可选）
                loss.backward()
                self.optimizer.step()
                gp_optimizer.step()
            else:
                self.model.train()
                self.likelihood.train()
                output = self.model(self.train_x)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
                loss = -mll(output, self.train_y)
                loss.backward()
                self.optimizer.step()

        self.model.eval()
        if self.likelihood:
            self.likelihood.eval()
        return self.model
    
def train_gp_models(train_x, train_y, use_adaptive_noise=True, num_iters=500):
    models = []
    likelihoods = []
    for i in range(train_y.shape[1]):
        trainer = GPTrainer(train_x, train_y[:, i], use_adaptive_noise=use_adaptive_noise, num_train_iters=num_iters)
        model = trainer.fit()
        models.append(model)
        likelihoods.append(trainer.likelihood)
    return models, likelihoods
