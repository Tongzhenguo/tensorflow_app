import argparse
import codecs
import json
import os
import shutil

from six import iteritems
from tensorflow.contrib.rnn import MultiRNNCell

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import time
import numpy as np
import tensorflow as tf

# Disable Tensorflow logging messages.
logging.getLogger('tensorflow').setLevel(logging.WARNING)


class CharRNN(object):
    """
    Character RNN model
    """

    def __init__(self, is_training, batch_size, num_unrollings, vocab_size,
                 hidden_size, max_grad_norm, embedding_size, num_layers,
                 learning_rate, model, dropout=0.0, input_dropout=0.0, use_batch=True):
        """
        类构造函数
        :param is_training: boolean,训练用还是测试用
        :param batch_size: int,批大小
        :param num_unrollings:int,要生成的训练数据段的数目
        :param vocab_size:int,嵌入词典大小
        :param hidden_size:int,隐层神经元个数
        :param max_grad_norm:float,最大梯度
        :param embedding_size:int,嵌入层向量维度
        :param num_layers:int,rnn网络层数
        :param learning_rate:float,学习速率
        :param model:string,RNN类型,枚举"rnn","lstm","gru"
        :param dropout:float,随机抛弃神经元的比例
        :param input_dropout:float,随机抛弃输入神经元的比例
        :param use_batch:是否分批训练
        """
        self.batch_size = batch_size
        self.num_unrollings = num_unrollings
        if not use_batch:
            self.batch_size = 1
            self.num_unrollings = 1
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_grad_norm = max_grad_norm
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.model = model
        self.dropout = dropout
        self.input_dropout = input_dropout
        if embedding_size <= 0:
            self.input_size = vocab_size
            # Don't do dropout on one hot representation.
            self.input_dropout = 0.0
        else:
            self.input_size = embedding_size
        self.model_size = (embedding_size * vocab_size +  # embedding parameters
                           # lstm parameters
                           4 * hidden_size * (hidden_size + self.input_size + 1) +
                           # softmax parameters
                           vocab_size * (hidden_size + 1) +
                           # multilayer lstm parameters for extra layers.
                           (num_layers - 1) * 4 * hidden_size *
                           (hidden_size + hidden_size + 1))

        # Placeholder to feed in input and targets/labels data.
        self.input_data = tf.placeholder(tf.int64,
                                         [self.batch_size, self.num_unrollings],
                                         name='inputs')
        self.targets = tf.placeholder(tf.int64,
                                      [self.batch_size, self.num_unrollings],
                                      name='targets')
        cell_fn = None
        if self.model == 'rnn':
            cell_fn = tf.contrib.rnn.BasicRNNCell
        elif self.model == 'lstm':
            cell_fn = tf.contrib.rnn.BasicLSTMCell
        elif self.model == 'gru':
            cell_fn = tf.contrib.rnn.GRUCell

        params = {}
        if self.model == 'lstm':
            # add bias to forget gate in lstm.
            params['forget_bias'] = 0.0
            params['state_is_tuple'] = True
        # Create multilayer cell.
        cell = cell_fn(
            self.hidden_size, reuse=tf.get_variable_scope().reuse,
            **params)

        cells = [cell]
        # more explicit way to create cells for MultiRNNCell than
        # [higher_layer_cell] * (self.num_layers - 1)
        for i in range(self.num_layers - 1):
            higher_layer_cell = cell_fn(
                self.hidden_size, reuse=tf.get_variable_scope().reuse,
                **params)
            cells.append(higher_layer_cell)

        if is_training and self.dropout > 0:
            cells = [tf.contrib.rnn.DropoutWrapper(
                cell,
                output_keep_prob=1.0 - self.dropout)
                     for cell in cells]

        multi_cell = MultiRNNCell(cells)

        with tf.name_scope('initial_state'):
            # zero_state is used to compute the intial state for cell.
            self.zero_state = multi_cell.zero_state(self.batch_size, tf.float32)
            # Placeholder to feed in initial state.
            self.initial_state = create_tuple_placeholders_with_default(
                multi_cell.zero_state(batch_size, tf.float32),
                extra_dims=(None,),
                shape=multi_cell.state_size)

        # Embeddings layers.
        with tf.name_scope('embedding_layer'):
            if embedding_size > 0:
                self.embedding = tf.get_variable(
                    'embedding', [self.vocab_size, self.embedding_size])
            else:
                self.embedding = tf.constant(np.eye(self.vocab_size), dtype=tf.float32)

            inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)
            if is_training and self.input_dropout > 0:
                inputs = tf.nn.dropout(inputs, 1 - self.input_dropout)

        with tf.name_scope('slice_inputs'):
            # Slice inputs into a list of shape [batch_size, 1] data colums.
            sliced_inputs = [tf.squeeze(input_, [1])
                             for input_ in tf.split(axis=1, num_or_size_splits=self.num_unrollings, value=inputs)]

        # Copy cell to do unrolling and collect outputs.
        outputs, final_state = tf.contrib.rnn.static_rnn(
            multi_cell, sliced_inputs,
            initial_state=self.initial_state)

        self.final_state = final_state

        with tf.name_scope('flatten_ouputs'):
            # Flatten the outputs into one dimension.
            flat_outputs = tf.reshape(tf.concat(axis=1, values=outputs), [-1, hidden_size])

        with tf.name_scope('flatten_targets'):
            # Flatten the targets too.
            flat_targets = tf.reshape(tf.concat(axis=1, values=self.targets), [-1])

        # Create softmax parameters, weights and bias.
        with tf.variable_scope('softmax'):
            softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size])
            softmax_b = tf.get_variable("softmax_b", [vocab_size])
            self.logits = tf.matmul(flat_outputs, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(self.logits)

        with tf.name_scope('loss'):
            # Compute mean cross entropy loss for each output.
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=flat_targets)
            self.mean_loss = tf.reduce_mean(loss)

        with tf.name_scope('loss_monitor'):
            # Count the number of elements and the sum of mean_loss
            # from each batch to compute the average loss.
            count = tf.Variable(1.0, name='count')
            sum_mean_loss = tf.Variable(1.0, name='sum_mean_loss')

            self.reset_loss_monitor = tf.group(sum_mean_loss.assign(0.0),
                                               count.assign(0.0),
                                               name='reset_loss_monitor')
            self.update_loss_monitor = tf.group(sum_mean_loss.assign(sum_mean_loss +
                                                                     self.mean_loss),
                                                count.assign(count + 1),
                                                name='update_loss_monitor')
            with tf.control_dependencies([self.update_loss_monitor]):
                self.average_loss = sum_mean_loss / count
                self.ppl = tf.exp(self.average_loss)

            # Monitor the loss.
            loss_summary_name = "average loss"
            ppl_summary_name = "perplexity"

            average_loss_summary = tf.summary.scalar(loss_summary_name, self.average_loss)
            ppl_summary = tf.summary.scalar(ppl_summary_name, self.ppl)

        # Monitor the loss.
        self.summaries = tf.summary.merge([average_loss_summary, ppl_summary],
                                          name='loss_monitor')

        self.global_step = tf.get_variable('global_step', [],
                                           initializer=tf.constant_initializer(0.0))

        self.learning_rate = tf.constant(learning_rate)
        if is_training:
            # learning_rate = tf.train.exponential_decay(1.0, self.global_step,
            #                                            5000, 0.1, staircase=True)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars),
                                              self.max_grad_norm)
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay_rate)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)

            self.train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                      global_step=self.global_step)

    def run_epoch(self, session, data_size, batch_generator, is_training,
                  verbose=0, freq=10, divide_by_n=1):
        """Runs the model on the given data for one full pass."""
        # epoch_size = ((data_size // self.batch_size) - 1) // self.num_unrollings
        epoch_size = data_size // (self.batch_size * self.num_unrollings)
        if data_size % (self.batch_size * self.num_unrollings) != 0:
            epoch_size += 1

        if verbose > 0:
            logging.info('epoch_size: %d', epoch_size)
            logging.info('data_size: %d', data_size)
            logging.info('num_unrollings: %d', self.num_unrollings)
            logging.info('batch_size: %d', self.batch_size)

        if is_training:
            extra_op = self.train_op
        else:
            extra_op = tf.no_op()

        # Prepare initial state and reset the average loss
        # computation.
        state = session.run(self.zero_state)
        self.reset_loss_monitor.run()
        start_time = time.time()
        for step in range(epoch_size // divide_by_n):
            # Generate the batch and use [:-1] as inputs and [1:] as targets.
            data = batch_generator.next()
            inputs = np.array(data[:-1]).transpose()
            targets = np.array(data[1:]).transpose()

            ops = [self.average_loss, self.final_state, extra_op,
                   self.summaries, self.global_step, self.learning_rate]

            feed_dict = {self.input_data: inputs, self.targets: targets,
                         self.initial_state: state}

            results = session.run(ops, feed_dict)
            average_loss, state, _, summary_str, global_step, lr = results

            ppl = np.exp(average_loss)
            if (verbose > 0) and ((step + 1) % freq == 0):
                logging.info("%.1f%%, step:%d, perplexity: %.3f, speed: %.0f words",
                             (step + 1) * 1.0 / epoch_size * 100, step, ppl,
                             (step + 1) * self.batch_size * self.num_unrollings /
                             (time.time() - start_time))

        logging.info("Perplexity: %.3f, speed: %.0f words per sec",
                     ppl, (step + 1) * self.batch_size * self.num_unrollings /
                     (time.time() - start_time))
        return ppl, summary_str, global_step

    def sample_seq(self, session, length, start_text, vocab_index_dict,
                   index_vocab_dict, temperature=1.0, max_prob=True):

        state = session.run(self.zero_state)

        # use start_text to warm up the RNN.
        if start_text is not None and len(start_text) > 0:
            seq = list(start_text)
            for char in start_text[:-1]:
                x = np.array([[char2id(char, vocab_index_dict)]])
                state = session.run(self.final_state,
                                    {self.input_data: x,
                                     self.initial_state: state})
            x = np.array([[char2id(start_text[-1], vocab_index_dict)]])
        else:
            vocab_size = len(vocab_index_dict.keys())
            x = np.array([[np.random.randint(0, vocab_size)]])
            seq = []

        for i in range(length):
            state, logits = session.run([self.final_state,
                                         self.logits],
                                        {self.input_data: x,
                                         self.initial_state: state})
            unnormalized_probs = np.exp((logits - np.max(logits)) / temperature)
            probs = unnormalized_probs / np.sum(unnormalized_probs)

            if max_prob:
                sample = np.argmax(probs[0])
            else:
                sample = np.random.choice(self.vocab_size, 1, p=probs[0])[0]

            seq.append(id2char(sample, index_vocab_dict))
            x = np.array([[sample]])
        return ''.join(seq)


class BatchGenerator(object):
    """Generate and hold batches."""

    def __init__(self, text, batch_size, n_unrollings, vocab_size,
                 vocab_index_dict, index_vocab_dict):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self.vocab_size = vocab_size
        self._n_unrollings = n_unrollings
        self.vocab_index_dict = vocab_index_dict
        self.index_vocab_dict = index_vocab_dict

        segment = self._text_size // batch_size

        # number of elements in cursor list is the same as
        # batch_size.  each batch is just the collection of
        # elements in where the cursors are pointing to.
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b] = char2id(self._text[self._cursor[b]], self.vocab_index_dict)
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
      the last batch of the previous array, followed by num_unrollings new ones.
      """
        batches = [self._last_batch]
        for step in range(self._n_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches


# Utility functions
def batches2string(batches, index_vocab_dict):
    """Convert a sequence of batches back into their (most likely) string
  representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, id2char_list(b, index_vocab_dict))]
    return s


def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (most likely) character representation."""
    return [id2char(c) for c in np.argmax(probabilities, 1)]


def char2id(char, vocab_index_dict):
    try:
        return vocab_index_dict[char]
    except KeyError:
        logging.info('Unexpected char %s', char)
        return 0


def id2char(index, index_vocab_dict):
    return index_vocab_dict[index]


def id2char_list(lst, index_vocab_dict):
    return [id2char(i, index_vocab_dict) for i in lst]


def create_tuple_placeholders_with_default(inputs, extra_dims, shape):
    if isinstance(shape, int):
        result = tf.placeholder_with_default(
            inputs, list(extra_dims) + [shape])
    else:
        subplaceholders = [create_tuple_placeholders_with_default(
            subinputs, extra_dims, subshape)
                           for subinputs, subshape in zip(inputs, shape)]
        t = type(shape)
        if t == tuple:
            result = t(subplaceholders)
        else:
            result = t(*subplaceholders)
    return result


def create_tuple_placeholders(dtype, extra_dims, shape):
    if isinstance(shape, int):
        result = tf.placeholder(dtype, list(extra_dims) + [shape])
    else:
        subplaceholders = [create_tuple_placeholders(dtype, extra_dims, subshape)
                           for subshape in shape]
        t = type(shape)

        # Handles both tuple and LSTMStateTuple.
        if t == tuple:
            result = t(subplaceholders)
        else:
            result = t(*subplaceholders)
    return result


def main():
    parser = argparse.ArgumentParser()

    # Data and vocabulary file
    parser.add_argument('--data_file', type=str,
                        default='data/tiny_shakespeare.txt',
                        help='data file')

    parser.add_argument('--encoding', type=str,
                        default='utf-8',
                        help='the encoding of the data file.')

    # Parameters for saving models.
    parser.add_argument('--output_dir', type=str, default='output',
                        help=('directory to store final and'
                              ' intermediate results and models.'))
    parser.add_argument('--n_save', type=int, default=1,
                        help='how many times to save the model during each epoch.')
    parser.add_argument('--max_to_keep', type=int, default=5,
                        help='how many recent models to keep.')

    # Parameters to configure the neural network.
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='size of RNN hidden state vector')
    parser.add_argument('--embedding_size', type=int, default=0,
                        help='size of character embeddings')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--num_unrollings', type=int, default=10,
                        help='number of unrolling steps.')
    parser.add_argument('--model', type=str, default='lstm',
                        help='which model to use (rnn, lstm or gru).')

    # Parameters to control the training.
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='minibatch size')
    parser.add_argument('--train_frac', type=float, default=0.9,
                        help='fraction of data used for training.')
    parser.add_argument('--valid_frac', type=float, default=0.05,
                        help='fraction of data used for validation.')
    # test_frac is computed as (1 - train_frac - valid_frac).
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout rate, default to 0 (no dropout).')

    parser.add_argument('--input_dropout', type=float, default=0.0,
                        help=('dropout rate on input layer, default to 0 (no dropout),'
                              'and no dropout if using one-hot representation.'))

    # Parameters for gradient descent.
    parser.add_argument('--max_grad_norm', type=float, default=5.,
                        help='clip global grad norm')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='initial learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate')

    # Parameters for logging.
    parser.add_argument('--log_to_file', dest='log_to_file', action='store_true',
                        help=('whether the experiment log is stored in a file under'
                              '  output_dir or printed at stdout.'))
    parser.set_defaults(log_to_file=False)

    parser.add_argument('--progress_freq', type=int,
                        default=100,
                        help=('frequency for progress report in training'
                              ' and evalution.'))

    parser.add_argument('--verbose', type=int,
                        default=0,
                        help=('whether to show progress report in training'
                              ' and evalution.'))

    # Parameters to feed in the initial model and current best model.
    parser.add_argument('--init_model', type=str,
                        default='',
                        help=('initial model'))
    parser.add_argument('--best_model', type=str,
                        default='',
                        help=('current best model'))
    parser.add_argument('--best_valid_ppl', type=float,
                        default=np.Inf,
                        help=('current valid perplexity'))

    # Parameters for using saved best models.
    parser.add_argument('--init_dir', type=str, default='',
                        help='continue from the outputs in the given directory')

    # Parameters for debugging.
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='show debug information')
    parser.set_defaults(debug=False)

    # Parameters for unittesting the implementation.
    parser.add_argument('--test', dest='test', action='store_true',
                        help=('use the first 1000 character to as data'
                              ' to test the implementation'))
    parser.set_defaults(test=False)

    args = parser.parse_args()

    # Specifying location to store model, best model and tensorboard log.
    args.save_model = os.path.join(args.output_dir, 'save_model/model')
    args.save_best_model = os.path.join(args.output_dir, 'best_model/model')
    args.tb_log_dir = os.path.join(args.output_dir, 'tensorboard_log/')
    args.vocab_file = ''

    # Create necessary directories.
    if args.init_dir:
        args.output_dir = args.init_dir
    else:
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        for paths in [args.save_model, args.save_best_model,
                      args.tb_log_dir]:
            os.makedirs(os.path.dirname(paths))

    # Specify logging config.
    if args.log_to_file:
        args.log_file = os.path.join(args.output_dir, 'experiment_log.txt')
    else:
        args.log_file = 'stdout'

    # Set logging file.
    if args.log_file == 'stdout':
        import sys
        logging.basicConfig(stream=sys.stdout,
                            format='%(asctime)s %(levelname)s:%(message)s',
                            level=logging.INFO,
                            datefmt='%I:%M:%S')
    else:
        logging.basicConfig(filename=args.log_file,
                            format='%(asctime)s %(levelname)s:%(message)s',
                            level=logging.INFO,
                            datefmt='%I:%M:%S')

    print('=' * 60)
    print('All final and intermediate outputs will be stored in %s/' % args.output_dir)
    print('All information will be logged to %s' % args.log_file)
    print('=' * 60 + '\n')

    if args.debug:
        logging.info('args are:\n%s', args)

    # Prepare parameters.
    if args.init_dir:
        with open(os.path.join(args.init_dir, 'result.json'), 'r') as f:
            result = json.load(f)
        params = result['params']
        args.init_model = result['latest_model']
        best_model = result['best_model']
        best_valid_ppl = result['best_valid_ppl']
        if 'encoding' in result:
            args.encoding = result['encoding']
        else:
            args.encoding = 'utf-8'
        args.vocab_file = os.path.join(args.init_dir, 'vocab.json')
    else:
        params = {'batch_size': args.batch_size,
                  'num_unrollings': args.num_unrollings,
                  'hidden_size': args.hidden_size,
                  'max_grad_norm': args.max_grad_norm,
                  'embedding_size': args.embedding_size,
                  'num_layers': args.num_layers,
                  'learning_rate': args.learning_rate,
                  'model': args.model,
                  'dropout': args.dropout,
                  'input_dropout': args.input_dropout}
        best_model = ''
    logging.info('Parameters are:\n%s\n', json.dumps(params, sort_keys=True, indent=4))

    # Read and split data.
    logging.info('Reading data from: %s', args.data_file)
    with codecs.open(args.data_file, 'r', encoding=args.encoding) as f:
        text = f.read()

    if args.test:
        text = text[:1000]
    logging.info('Number of characters: %s', len(text))

    if args.debug:
        n = 10
        logging.info('First %d characters: %s', n, text[:n])

    logging.info('Creating train, valid, test split')
    train_size = int(args.train_frac * len(text))
    valid_size = int(args.valid_frac * len(text))
    test_size = len(text) - train_size - valid_size
    train_text = text[:train_size]
    valid_text = text[train_size:train_size + valid_size]
    test_text = text[train_size + valid_size:]

    if args.vocab_file:
        vocab_index_dict, index_vocab_dict, vocab_size = load_vocab(
            args.vocab_file, args.encoding)
    else:
        logging.info('Creating vocabulary')
        vocab_index_dict, index_vocab_dict, vocab_size = create_vocab(text)
        vocab_file = os.path.join(args.output_dir, 'vocab.json')
        save_vocab(vocab_index_dict, vocab_file, args.encoding)
        logging.info('Vocabulary is saved in %s', vocab_file)
        args.vocab_file = vocab_file

    params['vocab_size'] = vocab_size
    logging.info('Vocab size: %d', vocab_size)

    # Create batch generators.
    batch_size = params['batch_size']
    num_unrollings = params['num_unrollings']
    train_batches = BatchGenerator(train_text, batch_size, num_unrollings, vocab_size,
                                   vocab_index_dict, index_vocab_dict)
    # valid_batches = BatchGenerator(valid_text, 1, 1, vocab_size,
    #                                vocab_index_dict, index_vocab_dict)
    valid_batches = BatchGenerator(valid_text, batch_size, num_unrollings, vocab_size,
                                   vocab_index_dict, index_vocab_dict)

    test_batches = BatchGenerator(test_text, 1, 1, vocab_size,
                                  vocab_index_dict, index_vocab_dict)

    if args.debug:
        logging.info('Test batch generators')
        logging.info(batches2string(train_batches.next(), index_vocab_dict))
        logging.info(batches2string(valid_batches.next(), index_vocab_dict))
        logging.info('Show vocabulary')
        logging.info(vocab_index_dict)
        logging.info(index_vocab_dict)

    # Create graphs
    logging.info('Creating graph')
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('training'):
            train_model = CharRNN(is_training=True, use_batch=True, **params)
        tf.get_variable_scope().reuse_variables()
        with tf.name_scope('validation'):
            valid_model = CharRNN(is_training=False, use_batch=True, **params)
        with tf.name_scope('evaluation'):
            test_model = CharRNN(is_training=False, use_batch=False, **params)
            saver = tf.train.Saver(name='checkpoint_saver', max_to_keep=args.max_to_keep)
            best_model_saver = tf.train.Saver(name='best_model_saver')

    logging.info('Model size (number of parameters): %s\n', train_model.model_size)
    logging.info('Start training\n')

    result = {}
    result['params'] = params
    result['vocab_file'] = args.vocab_file
    result['encoding'] = args.encoding

    try:
        # Use try and finally to make sure that intermediate
        # results are saved correctly so that training can
        # be continued later after interruption.
        with tf.Session(graph=graph) as session:
            graph_info = session.graph

            train_writer = tf.summary.FileWriter(args.tb_log_dir + 'train/', graph_info)
            valid_writer = tf.summary.FileWriter(args.tb_log_dir + 'valid/', graph_info)

            # load a saved model or start from random initialization.
            if args.init_model:
                saver.restore(session, args.init_model)
            else:
                tf.global_variables_initializer().run()
            for i in range(args.num_epochs):
                for j in range(args.n_save):
                    logging.info(
                        '=' * 19 + ' Epoch %d: %d/%d' + '=' * 19 + '\n', i + 1, j + 1, args.n_save)
                    logging.info('Training on training set')
                    # training step
                    ppl, train_summary_str, global_step = train_model.run_epoch(
                        session,
                        train_size,
                        train_batches,
                        is_training=True,
                        verbose=args.verbose,
                        freq=args.progress_freq,
                        divide_by_n=args.n_save)
                    # record the summary
                    train_writer.add_summary(train_summary_str, global_step)
                    train_writer.flush()
                    # save model
                    saved_path = saver.save(session, args.save_model,
                                            global_step=train_model.global_step)
                    logging.info('Latest model saved in %s\n', saved_path)
                    logging.info('Evaluate on validation set')

                    # valid_ppl, valid_summary_str, _ = valid_model.run_epoch(
                    valid_ppl, valid_summary_str, _ = valid_model.run_epoch(
                        session,
                        valid_size,
                        valid_batches,
                        is_training=False,
                        verbose=args.verbose,
                        freq=args.progress_freq)

                    # save and update best model
                    if (not best_model) or (valid_ppl < best_valid_ppl):
                        best_model = best_model_saver.save(
                            session,
                            args.save_best_model,
                            global_step=train_model.global_step)
                        best_valid_ppl = valid_ppl
                    valid_writer.add_summary(valid_summary_str, global_step)
                    valid_writer.flush()
                    logging.info('Best model is saved in %s', best_model)
                    logging.info('Best validation ppl is %f\n', best_valid_ppl)
                    result['latest_model'] = saved_path
                    result['best_model'] = best_model
                    # Convert to float because numpy.float is not json serializable.
                    result['best_valid_ppl'] = float(best_valid_ppl)
                    result_path = os.path.join(args.output_dir, 'result.json')
                    if os.path.exists(result_path):
                        os.remove(result_path)
                    with open(result_path, 'w') as f:
                        json.dump(result, f, indent=2, sort_keys=True)

            logging.info('Latest model is saved in %s', saved_path)
            logging.info('Best model is saved in %s', best_model)
            logging.info('Best validation ppl is %f\n', best_valid_ppl)
            logging.info('Evaluate the best model on test set')
            saver.restore(session, best_model)
            test_ppl, _, _ = test_model.run_epoch(session, test_size, test_batches,
                                                  is_training=False,
                                                  verbose=args.verbose,
                                                  freq=args.progress_freq)
            result['test_ppl'] = float(test_ppl)
    finally:
        result_path = os.path.join(args.output_dir, 'result.json')
        if os.path.exists(result_path):
            os.remove(result_path)
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, sort_keys=True)


def create_vocab(text):
    unique_chars = list(set(text))
    vocab_size = len(unique_chars)
    vocab_index_dict = {}
    index_vocab_dict = {}
    for i, char in enumerate(unique_chars):
        vocab_index_dict[char] = i
        index_vocab_dict[i] = char
    return vocab_index_dict, index_vocab_dict, vocab_size


def load_vocab(vocab_file, encoding):
    with codecs.open(vocab_file, 'r', encoding=encoding) as f:
        vocab_index_dict = json.load(f)
    index_vocab_dict = {}
    vocab_size = 0
    for char, index in iteritems(vocab_index_dict):
        index_vocab_dict[index] = char
        vocab_size += 1
    return vocab_index_dict, index_vocab_dict, vocab_size


def save_vocab(vocab_index_dict, vocab_file, encoding):
    with codecs.open(vocab_file, 'w', encoding=encoding) as f:
        json.dump(vocab_index_dict, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
