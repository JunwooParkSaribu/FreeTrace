from tensorflow.keras.models import load_model
import tfgraphviz as tfg
import tensorflow as tf

k_model = load_model('./models/reg_k_model.keras')
alpha_model = load_model('./models/reg_model_12.keras')
k_model.summary()
alpha_model.summary()
#tf.keras.utils.plot_model(k_model)
tf.keras.utils.plot_model(alpha_model)
