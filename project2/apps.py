from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
from keras.preprocessing.image import load_img
from predict import process_image, predict_class
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

item_names = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover',
              3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt',
              7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

app=Flask(__name__,template_folder='Template')
photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = './static/img'
configure_uploads(app, photos)

@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        image = load_img('./static/img/'+filename, grayscale=True, target_size=(28,28))
        image = process_image(image)
        pre_class, pre_prob = predict_class(image)
        pre_prob = "{:.6f}".format(pre_prob)
        answer = "For {} : <br>classified as class: {} with propability of: {}% ".format(filename, item_names[pre_class], pre_prob)
        return answer

    return render_template('upload.html')

app.run(host='0.0.0.0', port=5000, debug=True)