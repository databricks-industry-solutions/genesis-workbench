import torch


class MTLLoss(torch.nn.Module):
    """Args:
            losses: a list of task specific loss terms
            num_tasks: number of tasks
    """

    def __init__(self, num_tasks):
        super(MTLLoss, self).__init__()
        assert num_tasks > 1, "Number of tasks must be greater than 1"
        self.num_tasks = num_tasks
        self.log_sigma = torch.nn.Parameter(torch.zeros((num_tasks)))

    def get_precisions(self):
        return 0.5 * torch.exp(- 2.0 * self.log_sigma)

    def forward(self, loss_terms):
        assert loss_terms.numel() == self.num_tasks, f"Expected {self.num_tasks} loss terms, got {loss_terms.numel()}"

        total_loss = 0
        precisions = self.get_precisions()

        for task in range(self.num_tasks):
            total_loss += precisions[task] * loss_terms[task] + self.log_sigma[task]

        return total_loss
