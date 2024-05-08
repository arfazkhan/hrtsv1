from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect
import os
from datetime import datetime
from fpdf import FPDF
from flask import send_file


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///patients.db'
app.config['SECRET_KEY'] = 'hrts1'  # Change this to a random string
app.config['UPLOAD_FOLDER'] = 'static/pdf/'  # Change to pdf folder
db = SQLAlchemy(app)
csrf = CSRFProtect(app)

class Patient(db.Model):
    id = db.Column(db.String(20), primary_key=True)
    firstname = db.Column(db.String(50), nullable=False)
    lastname = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    dob = db.Column(db.String(20))  
    address = db.Column(db.String(200))
    city = db.Column(db.String(50))
    state = db.Column(db.String(50))
    zipcode = db.Column(db.String(10))
    bloodgroup = db.Column(db.String(10))
    gender = db.Column(db.String(10))
    doctor = db.Column(db.String(100))
    prevcondition = db.Column(db.Text)
    image = db.Column(db.String(100))
    medical_history_pdf = db.Column(db.String(100))  # Add this field
    

def generate_pdf(content, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Medical History", ln=True, align='C')
    pdf.cell(200, 10, txt=content, ln=True, align='L')
    pdf_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf.output(pdf_file_path)
    return pdf_file_path

@app.route('/')
def index():
    return render_template('add_patient.html')

@app.route('/patient/<patient_id>/')
def patient_profile(patient_id):
    patient = Patient.query.filter_by(id=patient_id).first()
    if patient:
        return render_template('patient_profile.html', patient=patient, calculate_age=calculate_age)
    else:
        return "Patient not found", 404

@app.route('/', methods=['POST'])
def add_patient():
    if request.method == 'POST':
        prefix = 'HRTS2024'
        last_patient = Patient.query.order_by(Patient.id.desc()).first()
        last_id = last_patient.id if last_patient else '000'
        new_id = prefix + str(int(last_id[-3:]) + 1).zfill(3)
        
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        dob = request.form['dob']
        address = request.form['address']
        city = request.form['city']
        state = request.form['state']
        zipcode = request.form['zipcode']
        bloodgroup = request.form['bloodgroup']
        gender = request.form['gender']
        doctor = request.form['doctor']
        prevcondition = request.form['prevcondition']
        image = request.files['patientimage']

        # Create the directories if they don't exist
        pdf_folder = os.path.join(app.static_folder, 'pdf')
        if not os.path.exists(pdf_folder):
            os.makedirs(pdf_folder)
        
        image_folder = os.path.join(app.static_folder, 'images')
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        # Rename and save the image
        image_ext = image.filename.rsplit('.', 1)[-1].lower()
        image_filename = f"{new_id}.{image_ext}"
        image_path = os.path.join(image_folder, image_filename)
        image.save(image_path)

        patient = Patient(id=new_id, firstname=firstname, lastname=lastname, email=email, phone=phone, dob=dob, 
                          address=address, city=city, state=state, zipcode=zipcode, bloodgroup=bloodgroup, 
                          gender=gender, doctor=doctor, prevcondition=prevcondition, image=image_filename)
        
        db.session.add(patient)
        db.session.commit()

        # Generate PDF
        pdf_file_path = generate_pdf(prevcondition, f"{new_id}_medical_history.pdf")

        # Save PDF file path to the database
        patient.medical_history_pdf = pdf_file_path
        db.session.commit()

        return redirect('/')

    
def calculate_age(dob):
    today = datetime.today()
    dob = datetime.strptime(dob, "%Y-%m-%d")
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return age

@app.route('/view_patients/')
def view_patients():
    patients = Patient.query.all()
    return render_template('view_patients.html', patients=patients, calculate_age=calculate_age)

@app.route('/download_history/<path:filename>')
def download_history(filename):
    # Construct the full path to the PDF file
    pdf_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Check if the file exists
    if os.path.exists(pdf_file_path):
        # Send the file as an attachment with the correct filename
        return send_file(pdf_file_path, as_attachment=True, download_name=filename)
    else:
        return "File not found", 404



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
