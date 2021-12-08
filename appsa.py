import os
import numpy as np
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import tensorflow as tf
import pandas as pd
import random
import matplotlib.pyplot as plt 

def viz_num(num):
    #Reshape the 768 values to a 28x28 image
    image = X_raw_final[num].reshape([28,28])
    fig = plt.figure(figsize=(1, 1))
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.axis("off")
    fig.show()
    return fig


MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model.h5')
if not os.path.isdir(MODEL_DIR):
    os.system('runipy train.ipynb')

model = load_model('C:/Users/User/Desktop/Demo_CNN/model.h5')
# st.markdown('<style>body{color: White; background-color: DarkSlateGrey}</style>', unsafe_allow_html=True)

st.title('My Digit Recognizer')
st.markdown('''
Try to write a digit!
''')



SIZE = 192
mode = st.checkbox("Draw (or Delete)?", True)
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas')

if canvas_result.image_data is not None:
    # img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    # rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    # st.write('Model Input')
    # st.image(rescaled)
    img = canvas_result.image_data

    image = Image.fromarray((img[:, :, 0]).astype(np.uint8))
    image = image.resize((28, 28))
    image = image.convert('L')
    image = (tf.keras.utils.img_to_array(image)/255)
    image = image.reshape(1,28,28,1)
    test_x = tf.convert_to_tensor(image)

if st.button('Predict'):
    #test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    val = model.predict(test_x)
    st.write(f'result: {np.argmax(val[0])}')  
    #st.bar_chart(val[0])


##Display button for prediction ###

#def load_data():
if st.selectbox('Tools', ['Randomizing Tool', 'Drawing Tool']) == 'Randomizing Tool':
   uploaded_file = st.file_uploader("Choose a file")
   if uploaded_file is not None:
        data_test = pd.read_csv(uploaded_file)
        X_raw_final = data_test.values
        X_test_final = data_test.values.reshape(data_test.shape[0], 28, 28, 1)
        pred_testing = model.predict(X_test_final)
        pred_testing = np.argmax(pred_testing, axis=1)
        #return  pred_testing

        if st.button('Predict a random image from our dataframe'):
            random_number = np.random.choice(28000)
            st.write('Picture number ' + str(random_number))
            st.write('Predicted number : ' + str(pred_testing[random_number]))
            viz = viz_num(random_number)
            st.pyplot(viz) 

#data = load_data()


### Visualization function ###
# def viz_num(num):
#     #Reshape the 768 values to a 28x28 image
#     image = X_raw_final[num].reshape([28,28])
#     fig = plt.figure(figsize=(1, 1))
#     plt.imshow(image, cmap=plt.get_cmap('gray'))
#     plt.axis("off")
#     fig.show()
#     return fig

# ### Display button for prediction ###
# if st.button('Predict a random image from our dataframe'):
#     random_number = np.random.choice(28000)
#     st.write('Picture number ' + str(random_number))
#     st.write('Predicted number : ' + str(pred_testing[random_number]))
#     viz = viz_num(random_number)
#     st.pyplot(viz) 
