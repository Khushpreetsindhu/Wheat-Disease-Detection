import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.utils import register_keras_serializable # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input # type: ignore

@register_keras_serializable()
class ChannelMean(layers.Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=-1, keepdims=True)

@register_keras_serializable()
class ChannelMax(layers.Layer):
    def call(self, inputs):
        return tf.reduce_max(inputs, axis=-1, keepdims=True)

# CBAM block 
def cbam_block(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    # CHANNEL ATTENTION
    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    max_pool = layers.GlobalMaxPooling2D()(input_feature)

    shared_dense = tf.keras.Sequential([
        layers.Dense(channel // ratio, activation='relu'),
        layers.Dense(channel)
    ])

    avg_out = shared_dense(avg_pool)
    max_out = shared_dense(max_pool)

    channel_attention = layers.Add()([avg_out, max_out])
    channel_attention = layers.Activation('sigmoid')(channel_attention)
    channel_attention = layers.Reshape((1, 1, channel))(channel_attention)
    channel_refined = layers.Multiply()([input_feature, channel_attention])

    # SPATIAL ATTENTION
    avg_pool_spatial = ChannelMean()(channel_refined)
    max_pool_spatial = ChannelMax()(channel_refined)
    concat = layers.Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])

    spatial_attention = layers.Conv2D(filters=1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    refined_feature = layers.Multiply()([channel_refined, spatial_attention])

    return refined_feature

#loading model 
@st.cache_resource
def load_efficientnet_cbam_model():
    return load_model("efficientnet_cbam_wheat.keras", custom_objects={
        "cbam_block": cbam_block,
        "ChannelMean": ChannelMean,
        "ChannelMax": ChannelMax
    })

model = load_efficientnet_cbam_model()
class_names = ['BlackPoint', 'FurasiumFootRot', 'HealthyLeaf', 'LeafBlight', 'WheatBlast'] 

#Streamlit 
st.title("Wheat Disease Classifier")
st.write("Upload an image to classify the disease.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224, 224))
    img_array = preprocess_input(np.array(img))  
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    prediction = tf.nn.softmax(prediction).numpy()
    st.write("Predicted probablities:", prediction)
    predicted_class = class_names[np.argmax(prediction)]
    st.success(f"**Predicted Class:** {predicted_class}")


