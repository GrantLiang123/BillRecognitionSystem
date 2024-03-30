from flask import Flask,jsonify,request
#from flask_cors import CORS
from bill_recognize_system.draw_picture.LedgerTable import create_table_image as f1




app=Flask(__name__)
#CORS(app)

@app.route('/MaNongBarbecue/001',methods=['GET','POST'])
def generate_image():
    data = request.json['data']
    img_buffer = f1(data)
    img_data = img_buffer.getvalue()
    return jsonify({'image_data': img_data.decode('latin-1')})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8809, debug=True)
