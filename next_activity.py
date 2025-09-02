import os
import argparse

import tensorflow as tf
from models.multiAttention import multi_att
from utils import loader
from utils import constants

from sklearn import metrics
import numpy as np

parser = argparse.ArgumentParser(description='Multi-Attention Prediction - Next Activity')

parser.add_argument('--dataset', required=True, type=str, help='dataset name')
parser.add_argument('--task', type=constants.Task, default=constants.Task.NEXT_ACTIVITY, help='task type')
parser.add_argument('--model_weight_dir', default='./modelWeight/', type=str, help='model directory')
parser.add_argument('--epochs', required=True, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')

parser.add_argument('--erEmbed', action='store_true', help='Use event relation graph embedding')
parser.add_argument('--pmEmbed', action='store_true', help='Use process model graph embedding')

args = parser.parse_args()
GPU = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(GPU[0], True)
print("GPUs Available:", tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


if __name__ == "__main__":
    model_weight_path = f"{args.model_weight_dir}/{args.dataset}"
    if not os.path.exists(model_weight_path):
        os.makedirs(model_weight_path)
    model_weight_path = f"{model_weight_path}/next_activity_ckpt.weights.h5"

    data_loader = loader.LogsDataLoader(name=args.dataset)

    (train_df, test_df, x_word_dict, y_word_dict,
     max_case_length, vocab_size, num_output) = data_loader.load_data(args.task)

    train_token_x, train_token_y = data_loader.prepare_data_next_activity(train_df, x_word_dict,
                                                                          y_word_dict, max_case_length)

    if args.erEmbed:
        erEmbed = np.load(f'./datasets/{args.dataset}/processed/er_rgcn_{args.dataset}.npy')
        print("Loading Event Relation embedding...")
        print("erEmbed name: ", f'er_gcn_{args.dataset}.npy')
        print("erEmbed shape: ", erEmbed.shape)
    else:
        erEmbed = None
        print("erEmbed is set to None.")

    if args.pmEmbed:
        pmEmbed = np.load(f'./datasets/{args.dataset}/processed/pm_wgat_{args.dataset}.npy')
        print("Loading process model graph embedding...")
        print("pmEmbed name: ", f'pm_gat_{args.dataset}.npy')
        print("pmEmbed shape: ", pmEmbed.shape)
    else:
        pmEmbed = None
        print("pmEmbed is set to None.")

    with tf.device('/GPU:0'):
        print("Loading multi_att...")
        model = multi_att.get_next_activity_model(
            max_case_length=max_case_length,
            vocab_size=vocab_size,
            output_dim=num_output,
            er_embed=erEmbed,
            pm_embed=pmEmbed
        )


    initial_learning_rate = 0.005
    decay_steps = 10000
    alpha = 0.001
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        alpha=alpha  # 最小学习率
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_weight_path, save_weights_only=True, monitor="sparse_categorical_accuracy", mode="max", save_best_only=True)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="sparse_categorical_accuracy", patience=20, mode="max",restore_best_weights=True)

    with tf.device('/GPU:0'):
        model.fit(train_token_x, train_token_y,
                epochs=args.epochs, batch_size=args.batch_size,
                shuffle=True, verbose=2, callbacks=[model_checkpoint_callback])

    k, accuracies, fscores, precisions, recalls = [], [], [], [], []
    for i in range(max_case_length):
        test_data_subset = test_df[test_df["k"] == i]
        if len(test_data_subset) > 0:
            test_token_x, test_token_y = data_loader.prepare_data_next_activity(test_data_subset, x_word_dict, y_word_dict, max_case_length)
            x_id_to_word = {v: k for k, v in x_word_dict.items()}
            y_id_to_word = {v: k for k, v in y_word_dict.items()}
            out = model.predict(test_token_x)
            y_pred = np.argmax(out, axis=1)

            accuracy = metrics.accuracy_score(test_token_y, y_pred)

            print("\nprefix-k: ", i+1)
            print("accuracy: ", accuracy)

            precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
                test_token_y, y_pred, average="weighted", zero_division=0)
            k.append(i)
            accuracies.append(accuracy)
            fscores.append(fscore)
            precisions.append(precision)
            recalls.append(recall)

    k.append(i + 1)
    accuracies.append(np.mean(accuracy))
    fscores.append(np.mean(fscores))
    precisions.append(np.mean(precisions))
    recalls.append(np.mean(recalls))
    print('\nAverage accuracy across all prefixes:', np.mean(accuracies))
    print('Average f-score across all prefixes:', np.mean(fscores))
    print('Average precision across all prefixes:', np.mean(precisions))
    print('Average recall across all prefixes:', np.mean(recalls))