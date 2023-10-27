import torch
from src.models.train_model import test_data_loader, train_data_loader, load_model_configs, train_model, save_model_results

def test_train_data_loader():
    # import from the dataset we processed
    train_dataset = torch.load("src/data/processed/train_dataset.pt")

    #checks if is instance of torch.utils.data.Dataset
    assert isinstance(train_dataset, torch.utils.data.Dataset)

    
    
    # Call the function to get a DataLoader object
    train_dataloader = train_data_loader(train_dataset)
    
    # Check that the returned object is a DataLoader
    assert isinstance(train_dataloader, torch.utils.data.DataLoader)
 
    print("test_train_data_loader Test passed!")

def test_load_model_configs():
    # Call the function to load the model and model configs
    model, optimizer, loss_fn = load_model_configs()
    
    # Check that the returned objects are not None
    assert model is not None
    assert optimizer is not None
    assert loss_fn is not None
    print("test_load_model_configs Test passed!")

def test_train_model():
    # Create a dummy model
    model = torch.nn.Linear(10, 2)
    
    # Create dummy data
    train_dataset = torch.utils.data.TensorDataset(torch.randn(10, 10), torch.randint(0, 2, (10,)))
    test_dataset = torch.utils.data.TensorDataset(torch.randn(10, 10), torch.randint(0, 2, (10,)))
    
    # Call the function to train the model
    _, train_loss, train_accuracy, val_loss, val_accuracy = train_model(model.__class__.__name__,
                                                                               model,
                                                                                 criterion=torch.nn.BCELoss(),
                                                                                   optimizer=torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9),
                                                                                     trainloader=train_data_loader(train_dataset),
                                                                                       valloader=test_data_loader(test_dataset),
                                                                                         epochs=1,
                                                                                           patience=1,
                                                                                             verbose=True)
    
    # Check that the returned values are not None
    assert train_loss is not None
    assert train_accuracy is not None
    assert val_loss is not None
    assert val_accuracy is not None
    print("test_train_model Test passed!")

def test_save_model_results():
    # Create dummy data
    train_loss = [1, 2, 3]
    train_accuracy = [0.5, 0.6, 0.7]
    val_loss = [4, 5, 6]
    val_accuracy = [0.8, 0.9, 1.0]
    model_name = "dummy_model"
    
    # Call the function to save the model results
    save_model_results(train_loss, train_accuracy, val_loss, val_accuracy, model_name)
    
    # Check that the file was created
    import os
    assert os.path.isfile(f"src/models/{model_name}_results.txt")
    print("test_save_model_results Test passed!")

if __name__ == "__main__":
    test_train_data_loader()
    test_load_model_configs()
    test_train_model()
    test_save_model_results()