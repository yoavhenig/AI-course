from flask import Flask
from flask_uploads import UploadSet, configure_uploads, IMAGES
from keras.preprocessing.image import load_img
from model import process_image, predict_class

app = Flask(__name__)
photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = './static/img'
configure_uploads(apps, photos)

@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        image = load_img('./static/img' + filename, grayscale=True, target_size=(28,28))
        image = process_image(image)
        pre_class, pre_prob = predict_class(image)
        answer = "For {} : <br>classified as class: {} with propability of: {.6f}% ".format(filename, pre_class, pre_prob)
        return answer

    return render_template('upload.html')

app.run(host='0.0.0.0', port=5000, debug=True)