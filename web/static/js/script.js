// -------------- Modal Open/Close Logic --------------
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

// Close modal when clicking outside modal-content
window.addEventListener('click', (event) => {
  if (event.target === kbModal) {
    kbModal.style.display = 'none';
  }
});

// -------------- Load Knowledge Bases --------------
function loadKnowledgeBases() {
  fetch('/kb_list')
    .then(response => response.json())
    .then(data => {
      const kbListEl = document.getElementById('kbList');
      const kbFolderSelect = document.getElementById('kb_folder');
      if (kbListEl) kbListEl.innerHTML = '';
      if (kbFolderSelect) kbFolderSelect.innerHTML = '';

      data.forEach(folderName => {
        // Create list item for KG with name and rename button
        const li = document.createElement('li');
        const nameSpan = document.createElement('span');
        nameSpan.textContent = folderName;
        nameSpan.classList.add("kb-name");

        // Implement single click with double click detection:
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

        // Populate dropdown for upload
        const option = document.createElement('option');
        option.value = folderName;
        option.textContent = folderName;
        kbFolderSelect.appendChild(option);
      });

      // Load files for the first KG by default if available
      if (data.length > 0) {
        showKBFiles(data[0]);
      } else {
        document.getElementById('filesContainer').innerHTML = '<p>No KG available.</p>';
      }
    })
    .catch(err => console.error('Error loading KG list:', err));
}

// -------------- Inline Rename --------------
function enableRename(li, oldName) {
  const nameSpan = li.querySelector('.kb-name');
  nameSpan.contentEditable = "true";
  nameSpan.focus();
  const finishRename = () => {
    nameSpan.contentEditable = "false";
    const newName = nameSpan.textContent.trim();
    if(newName && newName !== oldName) {
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
    if(evt.key === "Enter") {
      evt.preventDefault();
      nameSpan.blur();
    }
  });
}

function renameKB(oldName, newName) {
  return fetch('/rename_kb', {
    method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: new URLSearchParams({ old_name: oldName, new_name: newName })
  })
  .then(res => res.json())
  .then(resData => {
    if(resData.status !== 'success') {
      throw new Error(resData.message || 'Rename error');
    }
  });
}

// -------------- Show KG Files --------------
function showKBFiles(kbName) {
  const kbFolderSelect = document.getElementById('kb_folder');
  if(kbFolderSelect) kbFolderSelect.value = kbName;
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

// -------------- Delete File --------------
function deleteFile(kbName, filename) {
  return fetch('/delete_file', {
    method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: new URLSearchParams({ kb_folder: kbName, filename: filename })
  })
  .then(res => res.json())
  .then(resData => {
    if(resData.status !== 'success'){
      throw new Error(resData.message || 'Delete error');
    }
  });
}

// -------------- Drag & Drop and Multi-File Upload --------------
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
    updatePreview(e.target.files);
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
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const dt = new DataTransfer();
      for (let i = 0; i < files.length; i++) {
        dt.items.add(files[i]);
      }
      fileInput.files = dt.files;
      updatePreview(fileInput.files);
    }
  });
}

function updatePreview(fileList) {
  if (fileList && fileList.length > 0) {
    let content = "<ul>";
    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i];
      content += `<li><span class="file-icon">📄</span> ${file.name} (${file.size} bytes)</li>`;
    }
    content += "</ul>";
    previewBox.innerHTML = content;
  } else {
    previewBox.innerHTML = "<p>No file selected.</p>";
  }
}
