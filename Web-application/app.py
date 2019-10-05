from flask import Flask, render_template, Response, redirect, url_for, request, send_from_directory
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
import langdetect, io
from langdetect.lang_detect_exception import LangDetectException
import pickle, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
Bootstrap(app)
app.config['UPLOAD_FOLDER'] = BASE_DIR + '/files/uploads' 
app.config['RESULTS_FOLDER'] = BASE_DIR + '/files/results' 

lang_nb_model = pickle.load(open(os.path.join(BASE_DIR,'pkl_objects/language/nbmodel.pkl'),'rb'))
lang_bow_model_char = pickle.load(open(os.path.join(BASE_DIR,'pkl_objects/language/bowmodelchar.pkl'),'rb'))
ara_NB_model = pickle.load(open(os.path.join(BASE_DIR,'pkl_objects/sentiment/aranbmodel.pkl'), 'rb')) 
tun_NB_model = pickle.load(open(os.path.join(BASE_DIR,'pkl_objects/sentiment/tunnbmodel.pkl'), 'rb')) 
ara_bow_model = pickle.load(open(os.path.join(BASE_DIR,'pkl_objects/sentiment/arabowmodel.pkl'), 'rb')) 
tun_bow_model = pickle.load(open(os.path.join(BASE_DIR,'pkl_objects/sentiment/tunbowmodel.pkl'), 'rb')) 

def predict_language(input_text): 
    try:
        res = langdetect.detect_langs(input_text)  
        result = str(res[0]).split(':')[0]
        if result != "ar": 
            predicted_lang = "Other"
        else: 
            input_dtm = lang_bow_model_char.transform([input_text])
            predicted_lang = lang_nb_model.predict(input_dtm)
        
        if predicted_lang == "TUN": 
            predicted_lang = "TUN"
        elif predicted_lang == "ARA": 
            predicted_lang = "ARA"
        
    except LangDetectException:
        predicted_lang = "Unkown"
    
    return predicted_lang

def predict_sentiment(user_input, lang):
    if lang == 'ARA':
        dtm=ara_bow_model.transform([user_input])
        return ara_NB_model.predict(dtm) 
    elif lang == 'TUN':
        dtm=tun_bow_model.transform([user_input])
        return tun_NB_model.predict(dtm)


@app.route('/', methods=['POST', 'GET'])
def input_text(): 
    downloadfile = os.path.join(app.config['UPLOAD_FOLDER'], 'hello.txt')
    return render_template("text.html", textcard="form", pathfile=downloadfile)

@app.route('/text', methods=["POST"])
def submit_text(): 
    input_text = request.form['input-text']
    predicted_sent_result = ""
    if input_text == "":
        return redirect('/')
    predicted_lang= predict_language(input_text)
    if predicted_lang == "Other":
        predicted_sent_result = "None"
    else:
        predicted_sent = predict_sentiment(input_text, predicted_lang)
        for p in predicted_sent: 
            if p == 1:
                predicted_sent_result = "POS"
            else:
                predicted_sent_result = "NEG"

    return render_template("text.html", textcard="results", message=input_text, lang=predicted_lang, sent=predicted_sent_result)


@app.route('/upload', methods=['POST', 'GET'])
def upload_file(): 
    downloadfile = os.path.join(app.config['UPLOAD_FOLDER'], 'hello.txt')
    return render_template("file.html", textcard="form", pathfile=downloadfile)
  
   
@app.route('/file', methods=["POST"])
def submit_file(): 
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect('/')
        
        file = request.files['file']
        print(file)
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            print(filename)
            langs = ""
            sents = ""
            with io.open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'r', encoding='utf8') as f:
                print("File stored in uploads and loaded")
                for doc in f: 
                    predicted_lang = predict_language(doc)
                    langs = langs + predicted_lang + '\n'

                    if predicted_lang == "Other" or predicted_lang == "Unknown": 
                        sents = sents + "None" + '\n'
                    else: 
                        predicted_sent = predict_sentiment(doc, predicted_lang)
                        if predicted_sent == None: 
                            sens = sents + "\n"
                        else:
                            for p in predicted_sent: 
                                if p == 1:
                                    predicted_sent2 = "POS"
                                else:
                                    predicted_sent2 = "NEG"
                            sents = sents + predicted_sent2 + '\n'
                f.close()
            
            with io.open(os.path.join(app.config['RESULTS_FOLDER'], filename + "-lang"), 'w', encoding='utf8') as f:
                 f.write(langs)
                 f.close()
            
            with io.open(os.path.join(app.config['RESULTS_FOLDER'], filename + "-sent"), 'w', encoding='utf8') as f:
                 f.write(sents)
                 f.close()
            
            return render_template("file.html", filecard="results", langs=filename, sents=filename)


@app.route("/download-lang/<langs>", methods=['POST'])
def download_language(langs): 
    langs_result = ""
    with io.open(os.path.join(app.config['RESULTS_FOLDER'], langs + "-lang"), 'r', encoding='utf8') as f:
        for doc in f: 
            langs_result = langs_result + doc 
        f.close()
    return Response(
        langs_result,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=languages.csv"})

@app.route("/download-sent/<sents>", methods=['POST'])
def download_sentiment(sents): 
    sents_result = ""
    with io.open(os.path.join(app.config['RESULTS_FOLDER'], sents + "-sent"), 'r', encoding='utf8') as f:
        for doc in f: 
            sents_result = sents_result + doc 
        f.close()
    return Response(
        sents_result,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=sentiments.csv"})

@app.route("/review", methods=['POST'])
def review(): 
    lang_review = request.form.get("lang_review")
    sent_review = request.form["sent_review"]
    message = request.form["message"]
    lang_results = request.form["lang"]
    sent_results = request.form["sent"]
    
    with io.open(BASE_DIR + "/Collected data/language.csv", 'a+', encoding='utf8') as f:
        if lang_review == "ok":
            if lang_results != "Other": 
                if lang_results == "TUN":
                    f.write(message + "," + "TUN")
                else:
                    f.write(message + "," + "ARA")
                f.write("\n")

        elif lang_review == "no": 
            if lang_results != "Other":
                if lang_results == "TUN":
                    f.write(message + "," + "ARA")
                else:
                    f.write(message + "," + "TUN")
                f.write("\n")

        f.close()

    with io.open(BASE_DIR + "/Collected data/sentiment.csv", 'a+', encoding='utf8') as f:
        if sent_review == "ok":
            if sent_results != "None": 
                if sent_results == "POS":
                    f.write(message + "," + "POS")
                else:
                    f.write(message + "," + "NEG")
                f.write("\n")

        elif sent_review == "no": 
            if sent_results != "None": 
                if sent_results == "POS":
                    f.write(message + "," + "POS")
                else:
                    f.write(message + "," + "NEG")
                f.write("\n")

        f.close()
    
    return redirect('/')


@app.route('/aboutus')
def about_us(): 
    return render_template("aboutus.html", textcard="form")

if __name__ == '__main__':
	app.run(debug=True)