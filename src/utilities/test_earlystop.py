import torch
from src.utilities.earlystop import EarlyStopping

def test_earlystop():
    model = torch.nn.Linear(10, 1)
    earlystop = EarlyStopping(model_name='test_model', patience=2, verbose=True, delta=0.01)
    val_loss = 1.0
    for i in range(5):
        earlystop(val_loss, model)
        if earlystop.early_stop:
            break
        val_loss -= 0.1
    print(round(earlystop.best_score, 1))
    assert round(earlystop.best_score, 1) == -0.6
    print(round(earlystop.val_loss_min, 1))
    assert round(earlystop.val_loss_min, 1) == 0.6

    #assert for the save checkpoint
def test_save_checkpoint():
    model = torch.nn.Linear(10, 1)
    earlystop = EarlyStopping(model_name='test_model', patience=2, verbose=True, delta=0.01)
    val_loss = 1.0
    earlystop(val_loss, model)
    assert earlystop.val_loss_min == 1.0
    assert earlystop.checkpoint_path == 'src/data/models/'+'test_model'+'/checkpoint.pt'
    earlystop(val_loss-0.1, model)
    assert earlystop.val_loss_min == 0.9
    assert earlystop.checkpoint_path == 'src/data/models/'+'test_model'+'/checkpoint.pt'
    earlystop(val_loss-0.2, model)
    assert earlystop.val_loss_min == 0.8
    assert earlystop.checkpoint_path == 'src/data/models/'+'test_model'+'/checkpoint.pt'

if __name__ == '__main__':
    test_earlystop()
    test_save_checkpoint()

