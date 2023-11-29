import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import time, copy, gdown
import torch
# from util import classify, set_background


# load class names
labels = ['alligator-snapping-turtle', 'common-musk-turtle', 'cumberland-slider-terrapin', 'european-pond-turtle',
          'false-map-turtle', 'florida-red-bellied-cooter', 'map-turtle', 'mississippi-map-turtle', 'mud-turtle',
          'peninsula-cooter', 'razorback-musk-turtle', 'red-eared-slider-terrapin', 'river-cooter',
          'snake-necked-turtle', 'softshell-turtle', 'spotted-turtle', 'yellow-bellied-slider-terrapin']

# set_background('./bgs/bg5.png')
# set_background('./edited_surprised_turtle.JPEG')


st.set_page_config(
    page_title="Animal species classifier App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    
)




with st.sidebar:
    logo = Image.open('./bgs/RSPCAlogo3.png')
    st.image(logo, width=250)
    st.markdown('**Animal classifier demo v0.1**')
    st.info('This is an AI powered tool developed by RSPCA aimed at assisting '
            'in identifying whether an animal belongs to the invasive species in the UK.', icon="ℹ️")
    st.warning('The tool currently only supports classification of the following animals: Terrapins', icon="⚠️")
    st.info('For the full list of invasive animal species and other relevant government guidance please '
                'follow [this link](https://www.gov.uk/guidance/invasive-non-native-alien-animal-species-rules-in-england-and-wales).')
    st.link_button("Get guidance here", "https://www.gov.uk/guidance/invasive-non-native-alien-animal-species-rules-in-england-and-wales")

if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False

def predict():

    try:
        st.session_state.prediction_done = True
        # classify image
        results = model(image_glob)
        st.session_state.results = results

        st.success('The image has been successfully classified! View the results in the ***View predictions*** tab.', icon="✅")
        st.toast('Prediction done!')
        time.sleep(.5)
        st.balloons()
    except:
        st.error('Oops! Something went wrong. Please try again.', icon="🚨")



# set title
st.title('Terrapin classification')

# # set meme photo
# turtle_img = Image.open('./surprised_turtle_text.jpg').convert('RGB')
# st.image(turtle_img, use_column_width=True)


# set header
# st.header('Please upload an image of a turtle')


# load classifier
@st.cache_resource
def load_the_model(url="https://drive.google.com/file/d/1opNJhGAiW77X8F9W3lEltCi1qpzPltxI/view?usp=sharing"):
    # Create a database session object that points to the URL.
    output = "model.pt"
    gdown.download(url=url, output=output, quiet=False, fuzzy=True)
    path = "./" + output
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    return model


# model = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
# path = './yolov5/model.pt'
# model = load_the_model(path)
model = load_the_model()


# display image
image_glob = None


tab1, tab2, tab3 = st.tabs(["**Upload image**", "**View predictions**", "**Export results**"])

with tab1:

    # upload file
    file = st.file_uploader('Please upload your image of a turtle for classification.', type=['jpeg', 'jpg', 'png'])

    on = st.toggle('Show the image.', value = True)

    if file is not None:
        image = Image.open(file).convert('RGB')
        image_glob = copy.deepcopy(image)

    _, col2, __ = st.columns([5, 1, 5])
    trigger = col2.button('Make prediction', on_click=predict)
    
    if on and image_glob is not None:
        st.write('This is the image you have uploaded!')

        _, col2, __ = st.columns([2,4,2])
        col2.image(image_glob, use_column_width=True)




with tab2:
    coll1, coll2, coll3 = st.columns(3)
    my_list = []
    try:
        if st.session_state.prediction_done:
            # write classification
            results = st.session_state.results
            preds = results.xyxyn

            output_image = np.squeeze(results.render())
            _, coll2, __ = st.columns([2,4,2])
            coll2.image(output_image, use_column_width=True)

            st.markdown("#### The following species have been identified")
            for pred in preds[0][:]:
                species_name = labels[int(pred[-1])]
                probability = round(pred[-2].item() * 100, 1)
                # st.markdown(f"**{species_name}** *with a certainty of* **{probability}%**")
                my_list.append([species_name, probability])
            output_df = pd.DataFrame( my_list, columns=(["Species name", 'Confidence in prediction (%)']))
            st.table(output_df)


    except:
        pass



with tab3:
    blank_1, col1, blank_2 = st.columns([4,1.5,4])
    try:
        st.table(output_df)

        @st.cache_data
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(output_df)



        col1.download_button(
            label="Download output as CSV",
            data=csv,
            file_name='large_df.csv',
            mime='text/csv',
        )

    except:
        pass

    try:
        btn = col1.download_button(
            label="Download image",
            data=output_image,
            file_name="Predictions.png",
            mime="image/png"
        )
    except:
        pass
    # a = output_image.astype('uint8')
    # st.write(f'{a.shape}')
    # PIL_image = Image.fromarray(output_image.astype('uint8'), 'RGB')
    # col2.download_button(
    #     label="Download image",
    #     data=b,
    #     file_name="Predictions.jpg",
    #     mime="image/jpg"
    # )