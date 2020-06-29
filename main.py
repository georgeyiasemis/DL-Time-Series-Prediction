import torch

from torch.autograd import Variable
import numpy as np
#################################################################################
#                                                                               #
#################################################################################

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
#################################################################################
#                                                                               #
#################################################################################

class EarlyStopping():

    """

	Early stops the training if validation loss doesn't
    improve after a given patience.

	"""

    def __init__(self, path, patience=7, verbose=False, delta=0):
#     	"""

#     	Parameters
#     	----------
#     	path : str
#     		Path of where to save checkpoints.
#     	patience : int, optional
#     		Iterations to wait after last time validation loss improved.
# 			 The default is 7.
#     	verbose : bool, optional
#     		If True, prints a message for each validation loss improvement.
# 			 The default is False.
#     	delta : float, optional
#     		Minimum change in the monitored quantity to qualify as an improvement.
# 			 Positive. The default is 0.

#     	"""

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = - val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):

        '''Saves model when validation loss decrease.'''

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), self.path + 'checkpoint.pt')
        self.val_loss_min = val_loss


def train(model, optimizer, criterion, epochs, train_loader, val_loader, device, scheduler=None,
            valid_every_step=10, early_stopping=None, regularization=None):
# 	"""
# 	Trains and returns a trained model


# 	Parameters
# 	----------
# 	model : torch.nn.Module
# 		Model to be trained.
# 	optimizer : torch.optim
# 		Optimizer to update network's weights.
# 	criterion : torch.nn loss function
# 		Objective function to be optimised.
# 	epochs : int
# 		Number of epochs to train the model.
# 	train_loader : torch.utils.data.DataLoader
# 		Iterable with training data.
# 	val_loader : torch.utils.data.DataLoader
# 		Iterable with validation data.
# 	device : str
# 		{cuda, cpu}.
# 	scheduler :  torch.optim.lr_scheduler, optional
# 		Reduces the value of the optimizer's learning rate. The default is None.
# 	valid_every_step : int, optional
# 		Evaluate model on validation data. The default is 1.
# 	early_stopping : EarlyStopping object, optional
# 		Performs early stopping. The default is None.
# 	regularization : float, optional
# 		Regularization parameter. Positive. The default is None.

# 	Returns
# 	-------
# 	model : torch.nn.Module
# 		Trained model.
# 	losses : list
# 		Train Losses.
# 	val_losses : list
# 		Validation losses.

# 	"""


    losses = []
    val_losses = []

    for epoch in range(epochs):

        train_loss = 0.0
        for i, (X, y, label) in enumerate(train_loader):

            X = Variable(X).to(device)
            y = Variable(y).to(device)
            y = y.reshape(y.shape[0], y.shape[1], -1)

            label = Variable(label).to(device)

            # Clear gradient buffers because we don't want any gradient
            # from previous epoch to carry forward, dont want to cummulate gradients
            optimizer.zero_grad()

            # get output from the model, given the inputs
            outputs = model(X, y)

            loss = criterion(outputs.squeeze(), label.squeeze())

            # Regularization
            if regularization is not None:
                l2_reg = 0.0
                # l2_reg.requires_grad=True
                for W in model.parameters():
                    l2_reg = l2_reg + W.norm(2) ** 2

                loss += regularization * l2_reg

            loss.to(device)

            train_loss += loss.item()

            # get gradients w.r.t to parameters
            loss.backward()

            # update parameters
            optimizer.step()

            # iter += 1
        ########################################################################
        # VALIDATION                                                           #
        ########################################################################
        if epoch % valid_every_step == 0:
            valid_loss = 0.0
            with torch.no_grad():

                for (x, y, label) in val_loader:

                    valid_loss += eval(model, x, y, label, criterion, device)

            if early_stopping != None:
                early_stopping(valid_loss, model ) #/ len(val_loader.dataset), model)
                if early_stopping.early_stop:
                    return model, losses, val_losses

            val_losses.append(valid_loss )#/ len(val_loader.dataset))

        ########################################################################



        losses.append(train_loss)# / len(train_loader.dataset))
        print("Epoch: [{:^4}/{:^5}]. Train Loss: {:8f}. Validation Loss: {:8f}".format(epoch,
													   epochs, losses[-1], val_losses[-1] ))

        # Update lr if scheduler option
        if scheduler != None:
            scheduler.step()

    return model, losses, val_losses



def eval(model, X, y, y_true, criterion, device):
# 	"""
# 	Performs one validation step.

# 	Parameters
# 	----------
# 	model : torch.nn.Module
# 		.
# 	X : torch.Tensor
# 		Features of shape (batch_size, seq_len, input_len).
# 	y : torch.Tensor
# 		Past target features of shape (batch_size, seq_len, 1).
# 	y_true : torch.Tensor
# 		Target values of shape (batch_size, 1).
# 	criterion : torch.nn Loss
# 		Objective function.
# 	device : str
# 		{cuda, cpu}.



# 	"""

    x = Variable(X).to(device)
    y = Variable(y).to(device)
    label = Variable(y_true).to(device)
    outputs = model(x, y)
    loss = criterion(outputs.squeeze(), label.squeeze())

    return loss.item()

def compute_sharpe_ratio(pred, true):
    pred = pred.detach().cpu().numpy()
    true = true.cpu().numpy()
    y = np.diff(true.reshape(-1, 1), axis=0)
    returns = np.sign(np.diff(pred.reshape(-1, 1), axis=0)) * y

    sr = np.sqrt(252) * (returns.mean() / np.sqrt(returns.var()))

    return sr
