from flask import Flask,render_template,request
import main as m
import json

app = Flask(__name__)

@app.route('/',methods=['GET'])
def new():
    return render_template('new.html')

@app.route('/',methods=['POST'])
def predict():

    data = request.json
    data1 = data.get('text1')
    data2 = data.get('text2')
    
    data = m.tagged_data(data1,data2)
    m.model_training(data)
    pred = m.similarity_score(0)[0][1]
    
    output = {"similarity score": pred}
    # return render_template('new.html',statement=prediction_statement)
    return json.dumps("Similarity Score is: "+str(output['similarity score']))


#if __name__=='__main__':
    #app.run(debug = False,host='0.0.0.0')