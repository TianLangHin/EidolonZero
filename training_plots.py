import matplotlib.pyplot as plt
import re

match_setting = re.compile('Setting: ([a-z0-9.]+)')
match_training_step = re.compile('Training step: ([0-9]+)')
match_chosen_move = re.compile('Chosen move: ([a-h1-8]+)')
match_game_outcome = re.compile('Game outcome: (.*)')
match_vae_pt_accuracy = re.compile('VAE Piece Type Accuracy: ([0-9.]+)')
match_vae_iou_accuracy = re.compile('VAE Piece Location IoU Accuracy: ([0-9.]+)')
match_convnet_loss = re.compile('ConvNet epoch [0-9]+, average loss: ([0-9.]+)')
match_vae_loss = re.compile('VAE epoch [0-9]+, average loss: ([0-9.]+)')

def parse_logs(filename: str) -> dict:
    log_results = {}

    setting_name = ''
    training_step = 0
    with open(filename, 'rt') as f:
        for line in f:
            if (name := match_setting.match(line)) is not None:
                setting_name = name.group(1)
                log_results[setting_name] = []
            elif (step := match_training_step.match(line)) is not None:
                training_step = int(step.group(1))
                new_entry = {
                    'moves': [],
                    'losses': {
                        'convnet': [],
                        'vae': [],
                    },
                }
                log_results[setting_name].append(new_entry)
            elif (move := match_chosen_move.match(line)) is not None:
                move = move.group(1)
                log_results[setting_name][training_step - 1]['moves'].append(move)
            elif (outcome := match_game_outcome.match(line)) is not None:
                outcome = float(outcome.group(1))
                log_results[setting_name][training_step - 1]['outcome'] = outcome
            elif (pt_acc := match_vae_pt_accuracy.match(line)) is not None:
                pt_acc = float(pt_acc.group(1))
                log_results[setting_name][training_step - 1]['pt_acc'] = pt_acc
            elif (iou_acc := match_vae_iou_accuracy.match(line)) is not None:
                iou_acc = float(iou_acc.group(1))
                log_results[setting_name][training_step - 1]['iou_acc'] = iou_acc
            elif (convnet_loss := match_convnet_loss.match(line)) is not None:
                convnet_loss = float(convnet_loss.group(1))
                log_results[setting_name][training_step - 1]['losses']['convnet'].append(convnet_loss)
            elif (vae_loss := match_vae_loss.match(line)) is not None:
                vae_loss = float(vae_loss.group(1))
                log_results[setting_name][training_step - 1]['losses']['vae'].append(vae_loss)

    return log_results

def plot_logs(data: dict):
    for model_setting in data.keys():
        pt_accuracies = [reading['pt_acc'] for reading in data[model_setting]]
        iou_accuracies = [reading['iou_acc'] for reading in data[model_setting]]
        fig = plt.figure()
        plt.title(f'Defogger Prediction Accuracy ({model_setting} model)')
        plt.plot(range(1, len(pt_accuracies) + 1), pt_accuracies)
        plt.plot(range(1, len(iou_accuracies) + 1), iou_accuracies)
        plt.xticks(range(1, len(pt_accuracies) + 1))
        plt.xlabel('Training step')
        plt.xlabel('Accuracy')
        plt.legend(['Piece Type Accuracy', 'Piece Location IoU'])
        plt.savefig(f'vae-accuracy-{model_setting}.png')

if __name__ == '__main__':
    logs_data = parse_logs('logs.txt')
    plot_logs(logs_data)
