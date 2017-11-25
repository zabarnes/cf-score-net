import shutil

from flask import Flask, render_template, request, url_for
from werkzeug import secure_filename

from model.model import Model

model = Model()

app = Flask(__name__)

@app.route('/')
def index():
    print request.endpoint
    return render_template(
        'index.html'
    )

@app.route('/score', methods = ['POST'])
def score():
   if request.method == 'POST':
      f = request.files['file']
      print f
      # get GUID filename for img
      dest_filename = "imgs/"+f.filename
      f.save(secure_filename(f.filename))
      shutil.move(f.filename, "static/"+dest_filename)
      ## send f through pipeline
      #scores = car.score()
      return render_template('score.html',
            url = url_for('static', filename=dest_filename),
            score = 23)

if __name__=='__main__':
    app.run()