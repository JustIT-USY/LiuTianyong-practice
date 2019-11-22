import os
import tokenization
import pandas as pd

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

file_path = os.path.join('data', 'train.csv')
train_df = pd.read_csv(file_path, encoding='utf-8')
train_data = []
for index, train in enumerate(train_df.values):
    guid = 'train-%d' % index
    print(train)
    text_a = tokenization.convert_to_unicode(str(train[0]))
    # text_b = tokenization.convert_to_unicode(str(train[1]))
    label = str(train[1])
    train_data.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    print(train_data)