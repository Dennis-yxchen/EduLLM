import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent'))

from agent.main_function import run_simulation

from flask import Flask, request, redirect, url_for, render_template, send_from_directory, flash, jsonify
from werkzeug.utils import secure_filename



# Define folder paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
TEMPLATE_FOLDER = os.path.join(BASE_DIR, 'web', 'templates')
STATIC_FOLDER = os.path.join(BASE_DIR, 'web', 'static')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'docx','json'}

app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
app.config['UPLOADS_DIR'] = UPLOADS_DIR
app.secret_key = 'replace_with_a_secure_key'

def allowed_file(filename):
    """Return True if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def count_folders_and_files():
    """
    Count the number of KG folders (subdirectories in UPLOADS_DIR) and
    total files inside them (ignoring files directly under UPLOADS_DIR).
    """
    if not os.path.exists(UPLOADS_DIR):
        return 0, 0

    folder_count = 0
    file_count = 0
    for item in os.listdir(UPLOADS_DIR):
        subpath = os.path.join(UPLOADS_DIR, item)
        if os.path.isdir(subpath):
            folder_count += 1
            for f in os.listdir(subpath):
                if os.path.isfile(os.path.join(subpath, f)):
                    file_count += 1
    return folder_count, file_count

@app.route('/', methods=['GET'])
def index():
    """Main page: display KG stats."""
    kb_count, total_files = count_folders_and_files()
    return render_template('index.html', kb_count=kb_count, total_files=total_files)

@app.route('/kb_list', methods=['GET'])
def kb_list():
    """Return a JSON array of KG folder names."""
    if not os.path.exists(UPLOADS_DIR):
        return jsonify([])
    dirs = []
    for item in os.listdir(UPLOADS_DIR):
        subpath = os.path.join(UPLOADS_DIR, item)
        if os.path.isdir(subpath):
            dirs.append(item)
    return jsonify(dirs)

@app.route('/kb_files/<kb_folder>', methods=['GET'])
def kb_files(kb_folder):
    """Return a JSON array of files inside the specified KG folder."""
    safe_kb = secure_filename(kb_folder)
    kb_path = os.path.join(UPLOADS_DIR, safe_kb)
    if not os.path.exists(kb_path):
        return jsonify([])
    files = []
    for item in os.listdir(kb_path):
        fpath = os.path.join(kb_path, item)
        if os.path.isfile(fpath):
            files.append(item)
    return jsonify(files)

@app.route('/create_kb', methods=['POST'])
def create_kb():
    """Create a new KG folder."""
    kb_name = request.form.get('kb_name', '').strip()
    if not kb_name:
        flash("No Knowledge Base name provided.")
        return redirect(url_for('index'))
    safe_name = secure_filename(kb_name)
    kb_path = os.path.join(UPLOADS_DIR, safe_name)
    if not os.path.exists(kb_path):
        os.makedirs(kb_path)
        flash(f"Knowledge Base '{safe_name}' created successfully!")
    else:
        flash(f"Knowledge Base '{safe_name}' already exists!")
    return redirect(url_for('index'))

@app.route('/rename_kb', methods=['POST'])
def rename_kb():
    """
    Rename an existing KG folder.
    Expects 'old_name' and 'new_name' from POST data.
    """
    old_name = request.form.get('old_name', '').strip()
    new_name = request.form.get('new_name', '').strip()
    if not old_name or not new_name:
        return jsonify({"status": "error", "message": "Invalid folder names"}), 400
    old_path = os.path.join(UPLOADS_DIR, secure_filename(old_name))
    new_path = os.path.join(UPLOADS_DIR, secure_filename(new_name))
    if not os.path.exists(old_path):
        return jsonify({"status": "error", "message": "Old folder does not exist"}), 400
    if os.path.exists(new_path):
        return jsonify({"status": "error", "message": "New folder name already exists"}), 400
    try:
        os.rename(old_path, new_path)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/upload_file', methods=['POST'])
def upload_file_route():
    """
    Upload one or more files into the specified KG folder.
    """
    kb_folder = request.form.get('kb_folder', '')
    if not kb_folder:
        flash("No Knowledge Base folder specified.")
        return redirect(url_for('index'))
    kb_path = os.path.join(UPLOADS_DIR, secure_filename(kb_folder))
    if not os.path.exists(kb_path):
        flash(f"Knowledge Base '{kb_folder}' does not exist.")
        return redirect(url_for('index'))
    if 'file' not in request.files:
        flash('No file part in the request.')
        return redirect(url_for('index'))
    files = request.files.getlist('file')
    if not files or len(files) == 0:
        flash('No file selected.')
        return redirect(url_for('index'))
    for file in files:
        if file.filename == '':
            continue
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(kb_path, filename))
    flash(f"File(s) uploaded to Knowledge Base '{kb_folder}'!")
    return redirect(url_for('index'))

@app.route('/delete_file', methods=['POST'])
def delete_file():
    """
    Delete a file from a KG folder.
    Expects 'kb_folder' and 'filename' in POST data.
    """
    kb_folder = request.form.get('kb_folder', '').strip()
    filename = request.form.get('filename', '').strip()
    if not kb_folder or not filename:
        return jsonify({"status": "error", "message": "Invalid data"}), 400
    safe_kb = secure_filename(kb_folder)
    safe_file = secure_filename(filename)
    file_path = os.path.join(UPLOADS_DIR, safe_kb, safe_file)
    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": "File does not exist"}), 400
    try:
        os.remove(file_path)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/uploads/<kb_folder>/<filename>')
def serve_file(kb_folder, filename):
    """Serve a file from the specified KG folder."""
    safe_kb = secure_filename(kb_folder)
    safe_filename = secure_filename(filename)
    file_path = os.path.join(UPLOADS_DIR, safe_kb, safe_filename)
    if not os.path.exists(file_path):
        flash("File does not exist.")
        return redirect(url_for('index'))
    return send_from_directory(os.path.join(UPLOADS_DIR, safe_kb), safe_filename)


@app.route('/generate_test', methods=['POST'])
def generate_test():
    kg_folder = request.form.get('kg_folder', '').strip()
    prompt_file = request.form.get('prompt_file', '').strip()

    if not kg_folder or not prompt_file:
        return jsonify({"status": "error", "message": "KG folder or prompt file not selected"}), 400

    abs_kg_path = os.path.join(app.config['UPLOADS_DIR'], secure_filename(kg_folder))
    full_prompt_path = os.path.join(abs_kg_path, secure_filename(prompt_file))

    if not os.path.exists(full_prompt_path):
        return jsonify({"status": "error", "message": "Selected file does not exist"}), 400

    try:
        run_simulation(abs_kg_path, prompt_file)
        return jsonify({
            "status": "success",
            "message": f"Generation completed for {prompt_file} in {kg_folder}."
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# TODO: 2. handle the transition of data file to json 3. storage of the output data 4. display of the file for user download
if __name__ == '__main__':
    os.makedirs(app.config['UPLOADS_DIR'], exist_ok=True)
    app.run(debug=True, port=5002)
