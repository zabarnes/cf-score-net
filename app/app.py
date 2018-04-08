import shutil
import sqlite3

from flask import Flask, render_template, request, url_for, redirect
from werkzeug import secure_filename

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # f_object = request.files['file']
        # # get GUID filename for img
        # f_object.save(secure_filename(f_object.filename))
        # shutil.move(f_object.filename, "static/"+dest_filename)
        return redirect(url_for('view', id="test"))

@app.route('/search', methods=['POST'])
def search():
    if request.method == 'POST':
        #Get image ID we are searching for
        image_id = request.form.get('searchID',"")
        #Check database for existence

        #Return view with image id
        view_url = url_for('view') + '?id=' + image_id
        return redirect(view_url)

@app.route('/view', methods = ['GET'])
def view():
    if request.method == 'GET':
        image_id = request.args.get('id',"")
        image_url = "static/imgs/" + image_id + ".jpg"
        return render_template('view.html', url=image_url)

@app.route('/login', methods = ['POST'])
def login():
    #Render scoring completion of the user
    if request.method == 'POST':
        #Login page
        user = request.form.get('user')
        score_url = url_for('score', user=user)
        return redirect(score_url)

@app.route('/score/<string:user>', methods=['GET','POST'])
def score(user):
    #Render scoring completion of the user
    if request.method == 'GET':
        print "get",user
        # Render first login page
        return render_template('score.html', user=user)

    elif request.method == 'POST':
        print "POST",user
        # Add score to database
        # And render login database
        return render_template('score.html', user=user)

if __name__ == '__main__':
    app.run()
