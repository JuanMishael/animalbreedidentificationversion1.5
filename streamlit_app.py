import streamlit as st
import cv2
from PIL import Image
import numpy as np
import time



def main():
    # basic page configuration
    st.set_page_config(
        page_title="ABI",
        page_icon="🐾"
    )

    st.title("Animal Breed Identification")

    animal_chs = st.sidebar.selectbox("Select Animal", ("Guinea Pig","Hamster","Spider","Rabbit","Snake"))      # This is the side bar selection

    aimodel_chs = st.sidebar.selectbox("Select Model", ("Resnet50","SVM","Logreg","Ensemble SVM+Logreg"))
    # a function for uploading files
    def upload_file():
        uploaded_file_toplabel = f'What Breed of {animal_chs}?'
        uploaded_file = st.file_uploader( uploaded_file_toplabel, type=["jpg", "jpeg","png"])
        return uploaded_file

    # a function for using the camera
    def using_camera():
        uploaded_file_toplabel = f'What Breed of {animal_chs}?'
        captured_data = st.camera_input(uploaded_file_toplabel, key="camera_capture", disabled=False)
        return captured_data

    warning = st.warning('Please allow this page to access the camera', icon="⚠️")

    option = st.radio("Choose an option", ("Upload", "Camera"))
    # conditional statement for choosing to upload or using the camera

    if option == "Upload":
        captured_img = upload_file()
    else:
        captured_img = using_camera()

    c1, c2= st.columns(2)  # this gives us a two column, one for input and the other one is for the result
    if captured_img is not None:
        im= Image.open(captured_img)
        img= np.asarray(im)
        image= cv2.resize(img,(256, 256))
        img= np.expand_dims(img, 0)
        c1.header('Input Image')
        c1.image(im)

    if captured_img is not None:
        c2.header('Identified As:')
        identified_as = ''
        prob_perc = 0
        # model
        if animal_chs == "Guinea Pig":
            if aimodel_chs == "Resnet50":
                from Control.Guineapig.con_guineapig_resnet import gpResNet
                prediction = gpResNet(captured_img)
                result = prediction.predict_image()
                identified_as = result[0]
                prob_perc = result[1]
                
            elif aimodel_chs == "SVM":
                from Control.Guineapig.con_guineapig_SVM import gpSVM
                prediction = gpSVM(captured_img)
                result = prediction.predict_image()
                identified_as = result[0]
                prob_perc = result[1]

            elif aimodel_chs == "Logreg":
                from Control.Guineapig.con_guineapig_logreg import gpLogReg
                prediction = gpLogReg(captured_img)
                result = prediction.predict_image()
                identified_as = result[0]
                prob_perc = result[1]
            else:
                from Control.Guineapig.con_guineapig_ensemble import gpEnsemble
                prediction = gpEnsemble(captured_img)
                result = prediction.predict_image()
                identified_as = result[0]
                prob_perc = result[1]
            
        elif animal_chs == "Hamster":
            if aimodel_chs == "Resnet50":
                print(animal_chs,aimodel_chs)
                return 
            elif aimodel_chs == "SVM":
                print(animal_chs,aimodel_chs)
                return 
            elif aimodel_chs == "Logreg":
                print(animal_chs,aimodel_chs)
                return
            else:
                print(animal_chs,aimodel_chs)
                return
            
        elif animal_chs == "Spider":
            if aimodel_chs == "Resnet50":
                print(animal_chs,aimodel_chs)
                return 
            elif aimodel_chs == "SVM":
                print(animal_chs,aimodel_chs)
                return 
            elif aimodel_chs == "Logreg":
                print(animal_chs,aimodel_chs)
                return
            else:
                print(animal_chs,aimodel_chs)
                return
            
        elif animal_chs == "Rabbit":
            if aimodel_chs == "Resnet50":
                print(animal_chs,aimodel_chs)
                return 
            elif aimodel_chs == "SVM":
                print(animal_chs,aimodel_chs)
                return 
            elif aimodel_chs == "Logreg":
                print(animal_chs,aimodel_chs)
                return
            else:
                print(animal_chs,aimodel_chs)
                return
            
        elif animal_chs == "Snake":
            if aimodel_chs == "Resnet50":
                print(animal_chs,aimodel_chs)
                return 
            elif aimodel_chs == "SVM":
                print(animal_chs,aimodel_chs)
                return 
            elif aimodel_chs == "Logreg":
                print(animal_chs,aimodel_chs)
                return
            else:
                print(animal_chs,aimodel_chs)
                return


        c2.subheader(identified_as)
        c2.subheader("{:.2%}".format(prob_perc))
        # loading function
        # with st.spinner('Wait for it...'):
        #     time.sleep(10)
        st.success('Done!')



    # Footer
    hide_footer = """
    <style>
    a:link , a:visited{
        color: black;
        background-color: transparent;
    }
    .school{
        text-decoration: none;
    }
    a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: none;
    }

    .footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100vw;
    background-color: white;
    color: black;
    text-align: center;
    }
    </style>
    <div class="footer">
    <p><a class='school'style='display: block; text-align: center;' href="https://www.ama.edu.ph/" target="_blank">AMA University and Colleges</a>
    <a class='school'style='display: block; text-align: center;' href="" target="_blank">Contact Us ☎️</a>
    </p>
    </div>
    """

    # this will implement the markdown code in the website
    st.markdown(hide_footer, unsafe_allow_html= True)

if __name__== '__main__':
    main()