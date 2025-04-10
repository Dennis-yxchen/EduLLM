// -------------------- Global variable for accumulated files --------------------
let accumulatedFiles = [];

// -------------------- DOMContentLoaded: Load KG list on page load --------------------
document.addEventListener('DOMContentLoaded', () => {
  loadKnowledgeBases();
});

// -------------------- Modal Open/Close Logic --------------------
const knowledgeBaseBtn = document.getElementById('knowledgeBaseBtn');
const kbModal = document.getElementById('kbModal');
const closeSpan = kbModal ? kbModal.querySelector('.close') : null;

if (knowledgeBaseBtn && kbModal) {
  knowledgeBaseBtn.addEventListener('click', () => {
    kbModal.style.display = 'block';
    loadKnowledgeBases();
  });
}

if (closeSpan && kbModal) {
  closeSpan.addEventListener('click', () => {
    kbModal.style.display = 'none';
  });
}

window.addEventListener('click', (event) => {
  if (event.target === kbModal) {
    kbModal.style.display = 'none';
  }
});

// -------------------- Load Knowledge Bases for Modal and Test Generation --------------------
function loadKnowledgeBases() {
  fetch('/kb_list')
    .then(response => response.json())
    .then(data => {
      const kbListEl = document.getElementById('kbList');
      const kbFolderSelect = document.getElementById('kb_folder');
      const testKgSelect = document.getElementById('kg_select');

      if (kbListEl) kbListEl.innerHTML = '';
      if (kbFolderSelect) kbFolderSelect.innerHTML = '';
      if (testKgSelect) testKgSelect.innerHTML = '';

      data.forEach(folderName => {
        // Create list item with KG name and rename button.
        const li = document.createElement('li');
        const nameSpan = document.createElement('span');
        nameSpan.textContent = folderName;
        nameSpan.classList.add("kb-name");

        // Single/double click logic using a timer:
        let clickTimer;
        nameSpan.addEventListener("click", () => {
          if (clickTimer) return;
          clickTimer = setTimeout(() => {
            showKBFiles(folderName);
            clickTimer = null;
          }, 250);
        });
        nameSpan.addEventListener("dblclick", () => {
          if (clickTimer) {
            clearTimeout(clickTimer);
            clickTimer = null;
          }
          showKBFiles(folderName);
        });

        const renameBtn = document.createElement('button');
        renameBtn.textContent = '✏️';
        renameBtn.classList.add("rename-btn");
        renameBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          enableRename(li, folderName);
        });

        li.appendChild(nameSpan);
        li.appendChild(renameBtn);
        kbListEl.appendChild(li);

        // Populate the modal upload KG dropdown.
        const opt1 = document.createElement('option');
        opt1.value = folderName;
        opt1.textContent = folderName;
        if (kbFolderSelect) kbFolderSelect.appendChild(opt1);

        // Populate the test generation KG dropdown.
        const opt2 = document.createElement('option');
        opt2.value = folderName;
        opt2.textContent = folderName;
        if (testKgSelect) testKgSelect.appendChild(opt2);
      });

      // Load files for the first KG by default, if available.
      if (data.length > 0) {
        showKBFiles(data[0]);
        loadPromptFilesForTest(data[0]); // Also update test generation prompt file dropdown.
      } else {
        const container = document.getElementById('filesContainer');
        if (container) container.innerHTML = '<p>No KG available.</p>';
      }
    })
    .catch(err => console.error('Error loading KG list:', err));
}

// -------------------- Inline Rename --------------------
function enableRename(li, oldName) {
  const nameSpan = li.querySelector('.kb-name');
  nameSpan.contentEditable = "true";
  nameSpan.focus();
  const finishRename = () => {
    nameSpan.contentEditable = "false";
    const newName = nameSpan.textContent.trim();
    if (newName && newName !== oldName) {
      renameKB(oldName, newName).then(() => {
        loadKnowledgeBases();
      }).catch(err => {
        alert('Rename failed: ' + err.message);
        nameSpan.textContent = oldName;
      });
    } else {
      nameSpan.textContent = oldName;
    }
  };
  nameSpan.addEventListener("blur", finishRename, { once: true });
  nameSpan.addEventListener("keydown", (evt) => {
    if (evt.key === "Enter") {
      evt.preventDefault();
      nameSpan.blur();
    }
  });
}

function renameKB(oldName, newName) {
  return fetch('/rename_kb', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ old_name: oldName, new_name: newName })
  })
  .then(res => res.json())
  .then(resData => {
    if (resData.status !== 'success') {
      throw new Error(resData.message || 'Rename error');
    }
  });
}

// -------------------- Show KG Files --------------------
function showKBFiles(kbName) {
  const kbFolderSelect = document.getElementById('kb_folder');
  if (kbFolderSelect) kbFolderSelect.value = kbName;
  fetch(`/kb_files/${encodeURIComponent(kbName)}`)
    .then(response => response.json())
    .then(files => {
      const filesContainer = document.getElementById('filesContainer');
      filesContainer.innerHTML = '';
      if (files.length === 0) {
        filesContainer.innerHTML = '<p>No files in this KG.</p>';
      } else {
        files.forEach(file => {
          const fileItem = document.createElement('div');
          fileItem.classList.add('file-item');

          const iconSpan = document.createElement('span');
          iconSpan.classList.add('file-icon');
          iconSpan.textContent = '📄';

          const nameSpan = document.createElement('span');
          nameSpan.classList.add('file-name');
          nameSpan.textContent = file;

          const delBtn = document.createElement('button');
          delBtn.classList.add('delete-btn');
          delBtn.textContent = '×';
          delBtn.addEventListener('click', () => {
            deleteFile(kbName, file)
              .then(() => showKBFiles(kbName))
              .catch(err => alert('Could not delete file: ' + err.message));
          });

          fileItem.appendChild(iconSpan);
          fileItem.appendChild(nameSpan);
          fileItem.appendChild(delBtn);
          filesContainer.appendChild(fileItem);
        });
      }
    })
    .catch(err => console.error('Error fetching files for', kbName, err));
}

// -------------------- Delete File --------------------
function deleteFile(kbName, filename) {
  return fetch('/delete_file', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ kb_folder: kbName, filename: filename })
  })
  .then(res => res.json())
  .then(resData => {
    if (resData.status !== 'success') {
      throw new Error(resData.message || 'Delete error');
    }
  });
}

// -------------------- Load Prompt Files for Test Generation --------------------
function loadPromptFilesForTest(kgName) {
  fetch(`/kb_files/${encodeURIComponent(kgName)}`)
    .then(response => response.json())
    .then(files => {
      const promptFileSelect = document.getElementById('prompt_file');
      if (promptFileSelect) {
        promptFileSelect.innerHTML = '';
        if (files.length === 0) {
          const opt = document.createElement('option');
          opt.value = "";
          opt.textContent = "No files available";
          promptFileSelect.appendChild(opt);
        } else {
          files.forEach(file => {
            const opt = document.createElement('option');
            opt.value = file;
            opt.textContent = file;
            promptFileSelect.appendChild(opt);
          });
        }
      }
    })
    .catch(err => console.error('Error loading prompt files:', err));
}

// Update prompt file dropdown when KG selection in test generation changes.
const testKgSelect = document.getElementById('kg_select');
if (testKgSelect) {
  testKgSelect.addEventListener('change', (e) => {
    const selectedKg = e.target.value;
    loadPromptFilesForTest(selectedKg);
  });
}

// -------------------- Test Generation Form Submission --------------------
const testGenForm = document.getElementById('testGenForm');
if (testGenForm) {
  testGenForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const kg_folder = document.getElementById('kg_select').value;
    const prompt_file = document.getElementById('prompt_file').value;
    if (!kg_folder || !prompt_file) {
      alert("Please select both a KG and a prompt file.");
      return;
    }
    const formData = new URLSearchParams();
    formData.append('kg_folder', kg_folder);
    formData.append('prompt_file', prompt_file);

    fetch('/generate_test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: formData.toString()
    })
    .then(res => res.json())
    .then(data => {
      const genResult = document.getElementById('genResult');
      if (data.status === 'success') {
        genResult.innerHTML = `<p><strong>Generated Question(s):</strong> ${JSON.stringify(data.generated_question)}</p>`;
      } else {
        genResult.innerHTML = `<p>Error: ${data.message}</p>`;
      }
    })
    .catch(err => {
      alert('Generation failed: ' + err.message);
    });
  });
}

// -------------------- Drag & Drop and Multi-File Upload --------------------
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => e.preventDefault());

const uploadWindow = document.getElementById('uploadWindow');
const fileInput = document.getElementById('fileInput');
const previewBox = document.getElementById('previewBox');

if (uploadWindow && fileInput && previewBox) {
  uploadWindow.addEventListener('click', () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', (e) => {
    let newFiles = Array.from(e.target.files);
    accumulatedFiles = accumulatedFiles.concat(newFiles);
    updateFileInput();
    updatePreview(accumulatedFiles);
  });

  uploadWindow.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadWindow.classList.add('dragover');
  });
  uploadWindow.addEventListener('dragleave', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadWindow.classList.remove('dragover');
  });
  uploadWindow.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadWindow.classList.remove('dragover');
    let droppedFiles = Array.from(e.dataTransfer.files);
    accumulatedFiles = accumulatedFiles.concat(droppedFiles);
    updateFileInput();
    updatePreview(accumulatedFiles);
  });
}

function updateFileInput() {
  const dt = new DataTransfer();
  accumulatedFiles.forEach(file => {
    dt.items.add(file);
  });
  fileInput.files = dt.files;
}

function updatePreview(fileList) {
  if (fileList && fileList.length > 0) {
    let content = "<ul>";
    fileList.forEach(file => {
      content += `<li><span class="file-icon">📄</span> ${file.name} (${file.size} bytes)</li>`;
    });
    content += "</ul>";
    previewBox.innerHTML = content;
  } else {
    previewBox.innerHTML = "<p>No file selected.</p>";
  }
}
