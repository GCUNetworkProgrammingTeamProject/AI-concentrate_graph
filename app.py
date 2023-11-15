from flask import Flask, render_template, request, jsonify
from concentrate_score import call_concentrate, concentrate
from chatgpt import chatgpt
# import mariadb
import sys
import threading

# def connect():
#     # Connect to MariaDB Platform
#     try:
#         conn = mariadb.connect(
#             user="dbid232",
#             password="dbpass232",
#             host="localhost",
#             port=3306,
#             database="db23209"
#         )
#     except mariadb.Error as e:
#         print(f"Error connecting to MariaDB Platform: {e}")
#         sys.exit(1)
#     return conn

################### 해야함 ###############################
# 함수 만들어서 concentrate() 함수 호출하고 디비에 값 넣기
# def run_concentrate():
#     result = concentrate()
#     # result = {1: 0.659, 2: 0.8448000000000002, 3: 0.6412, 4: 0.81996, 5: 0.7191200000000002, 6: 0.6866000000000001, 7: 0.8118800000000002, 8: 0.7954000000000001, 9: -1.32, 10: -2.6, 11: -2.6, 12: -0.7135199999999999, 13: 0.6328, 14: 0.7442000000000002, 15: 0.7287600000000001, 16: 0.7794800000000002}
#     conn = connect()
#     cursor = conn.cursor()
#     try:
#         print(result)
#         for key, val in result.items():
#             cursor.execute(
#                 "INSERT INTO tb_video_analysis_detail (video_analysis_seq, timeline, concentration) VALUES (?, ?, ?)",
#                 (7,key,val)
#             )
#     except mariadb.Error as e:
#         print(e)
    
#     conn.commit() 
#     conn.close()

# def run_chatgpt():
#     content = "오늘의 날씨는 어때" # get content from user
#     result = chatgpt(content)
#     conn = connect()
#     cursor = conn.cursor()
#     try:
#         cursor.execute(
#             "INSERT INTO tb_chatbot_log_detail (chatbot_log_detail_seq, chatbot_log_seq, question, answer) VALUES (?, ?, ?, ?)",
#             (1,7,content,result)
#         )
#     except mariadb.Error as e:
#         print(e)
    
#     conn.commit()
#     conn.close()


########################################################################################################
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Render the index.html template

@app.route('/concentrate')
def send_concentrate(url):
    url = request.args.get('url')
    data = concentrate(url)
    return jsonify(data)

@app.route('/chatgpt')
def send_chatgpt(url):
    q = request.args.get('q')
    data = chatgpt(q)
    return jsonify(data)

'''
@app.route('/concentrate')
def show_concentrate():
    # data = request.args.get('data')
    thread = threading.Thread(target=run_concentrate, args=())
    thread.start()
    return f"model started"

    

@app.route('/chatgpt')
def show_chatgpt():
    # data = request.args.get('data')
    thread = threading.Thread(target=run_chatgpt, args=())
    thread.start()
    return f"chatgpt started"
'''


'''
@app.route('/showconcentrate')
def show_graph():
    video_analysis_detail_seq = 1
    video_analysis_seq = 1
    data = {}
    conn = connect()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT timeline, concentration FROM tb_video_analysis_detail WHERE video_analysis_detail_seq = ? and video_analysis_seq = ?",
            (video_analysis_detail_seq,video_analysis_seq)
        )
        while True:
            row = cursor.fetchone()
            if row == None:
                break
            timeline, concentration = row[0], row[1]
            data[timeline] = concentration
    except mariadb.Error as e:
        print(e)
        return f"no data"
    
    conn.commit() 
    conn.close()
    return f"concentration data: {data}"

@app.route('/showchatgpt')
def show_chatgpt():
    chatbot_log_detail_seq = 1
    chatbot_log_seq = 1
    data = {}
    conn = connect()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT question, answer FROM tb_chatbot_log_detail WHERE chatbot_log_detail_seq = ? and chatbot_log_seq = ?",
            (chatbot_log_detail_seq, chatbot_log_seq)
        )
        while True:
            row = cursor.fetchone()
            if row == None:
                break
            question, answer = row[0], row[1]
            data[question] = answer
    except mariadb.Error as e:
        print(e)
        return f"no data"
    
    conn.commit() 
    conn.close()
    return f"chatgpt data: {data}"
'''

if __name__ == '__main__':
    app.run('0.0.0.0', port=7000, debug=True)