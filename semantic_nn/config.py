
EMBEDDING_DIM = 150

MAX_SEQUENCE_LENGTH = 20
VALIDATION_SPLIT = 0.2

RATE_DROP_LSTM = 0.17
RATE_DROP_DENSE = 0.25
NUMBER_LSTM = 150
NUMBER_DENSE_UNITS = 150
ACTIVATION_FUNCTION = 'tanh'

siamese_config = {
	'EMBEDDING_DIM': EMBEDDING_DIM,
	'MAX_SEQUENCE_LENGTH' : MAX_SEQUENCE_LENGTH,
	'VALIDATION_SPLIT': VALIDATION_SPLIT,
	'RATE_DROP_LSTM': RATE_DROP_LSTM,
	'RATE_DROP_DENSE': RATE_DROP_DENSE,
	'NUMBER_LSTM': NUMBER_LSTM,
	'NUMBER_DENSE_UNITS': NUMBER_DENSE_UNITS,
	'ACTIVATION_FUNCTION': ACTIVATION_FUNCTION
}