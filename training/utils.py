import torch, os
import matplotlib.pyplot as plt

def get_least_used_gpu():
    # Get available memory for each GPU
    gpu_free_memory = []
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        gpu_free_memory.append(free)

    least_used_gpu = max(range(torch.cuda.device_count()), key=lambda i: gpu_free_memory[i])
    return least_used_gpu

def get_top_available_gpus(n=3):
    # Get available memory for each GPU and return the indices of the top n GPUs
    gpu_free_memory = []
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        gpu_free_memory.append((free, i))
    top_gpus = sorted(gpu_free_memory, reverse=True)[:n]
    return [gpu_idx for free, gpu_idx in top_gpus]

def plot_graph(training_losses, training_accuracies, evaluation_accuracies, param, end_plot, figure_path=None, start_plot=0):

    DPI = 120
    FIGURE_SIZE_PIXEL = [2490, 1490]
    FIGURE_SIZE = [fsp / DPI for fsp in FIGURE_SIZE_PIXEL]

    if not (start_plot < end_plot):
        raise ValueError("end_plot must be greater than or equal to start_plot")
    
    # Calculate average loss
    total_loss = 0
    average_losses = []
    for index, loss in training_losses:
        total_loss += loss
        average_loss = total_loss / index
        average_losses.append([index, average_loss])
    
    # Filter the lists
    training_losses = training_losses[start_plot:end_plot]
    average_losses = average_losses[start_plot:end_plot]
    training_accuracies = training_accuracies[start_plot:end_plot]
    evaluation_accuracies = evaluation_accuracies[start_plot:end_plot]

    # Plot Training Loss
    plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    
    loss_x, loss_y = zip(*training_losses)
    average_loss_x, average_loss_y = zip(*average_losses)
    plt.scatter(loss_x, loss_y, color='blue', label='Training Loss')
    plt.plot(average_loss_x, average_loss_y, color='cyan', linestyle='-', label='Average Training Loss')
    
    plt.title("Training Loss")
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    plt.legend()

    # Annotate the loss every weight_saving_step
    for x, y in training_losses:
        if x % param['Weight_Saving_Step'] == 0:
            plt.annotate(f'{y:.4f}', xy=(x, y))
    last_epoch, last_loss = training_losses[-1]
    if last_epoch % param['Weight_Saving_Step'] != 0:
        plt.annotate(f'{last_loss:.4f}', xy=(last_epoch, last_loss))

    epoch_idx, min_loss_val = min(training_losses, key=lambda x: x[1])
    plt.text(
        0.01, 0.08,
        f"batch_size = {param['Batch_Size']}\nlr = {param['Learning_Rate']}\nepochs = {param['Num_Epochs']}\nlowest_loss = {min_loss_val:.4f} (epoch {epoch_idx})",
        transform=plt.gca().transAxes,
        verticalalignment='top',
        fontsize=10,
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3')
    )

    if figure_path is not None:
        loss_figure_path = os.path.join(figure_path, 'Training_loss.png')
        plt.savefig(loss_figure_path)
        plt.close()

    else:
        plt.show()

    # Plot Accuracy
    plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    
    ## Training  Accuracy
    accuracy_x, accuracy_y = zip(*training_accuracies)
    plt.plot(accuracy_x, accuracy_y, color='cyan', linestyle='-', marker='o', label='Training Accuracy')

    # Annotate the accuracy every weight_saving_step
    for x, y in training_accuracies:
        if x % param['Weight_Saving_Step'] == 0:
            plt.annotate(f'{y:.4f}', xy=(x, y))
    last_epoch, last_accuracy = training_accuracies[-1]
    if last_epoch % param['Weight_Saving_Step'] != 0:
        plt.annotate(f'{last_accuracy:.4f}', xy=(last_epoch, last_accuracy))

    ## Evaluation  Accuracy
    accuracy_x, accuracy_y = zip(*evaluation_accuracies)
    plt.plot(accuracy_x, accuracy_y, color='red', linestyle='-', marker='o', label='Evaluation Accuracy')

    # Annotate the accuracy every weight_saving_step
    for x, y in evaluation_accuracies:
        if x % param['Weight_Saving_Step'] == 0:
            plt.annotate(f'{y:.4f}', xy=(x, y))
    last_epoch, last_accuracy = evaluation_accuracies[-1]
    if last_epoch % param['Weight_Saving_Step'] != 0:
        plt.annotate(f'{last_accuracy:.4f}', xy=(last_epoch, last_accuracy))

    plt.title(f"Accuracy")
    plt.xlabel("Epoches")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    if figure_path is not None:
        accuracy_figure_path = os.path.join(figure_path, 'Accuracy.png')
        plt.savefig(accuracy_figure_path)
        plt.close()
    
    else:
        plt.show()