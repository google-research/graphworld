from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class LinearGCNModel(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class LinearGCN():
    def __init__(self, num_features, num_classes, hidden_channels):
        self._model = LinearGCNModel(num_features, num_classes, hidden_channels)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.01, weight_decay=5e-4)
        self._criterion = torch.nn.CrossEntropyLoss()

    def train_step(data):
        self._model.train()
        pass

    def train_step(data):
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x, data.edge_index)  # Perform a single forward pass.
        loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss

    def test(data):
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
        return test_acc

    def train(epochs, data):
        losses = {}
        for epoch in range(epochs):
            losses['{0:05}'.format(epoch+1)] = train(data)

        return losses
