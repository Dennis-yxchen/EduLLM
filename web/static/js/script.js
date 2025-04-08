// -------------------- Modal Logic --------------------
const knowledgeBaseBtn = document.getElementById('knowledgeBaseBtn');
const kbModal = document.getElementById('kbModal');
const closeSpan = kbModal ? kbModal.querySelector('.close') : null;

if (knowledgeBaseBtn && kbModal) {
  // Open modal on click
  knowledgeBaseBtn.addEventListener('click', () => {
    kbModal.style.display = 'block';
    loadKnowledgeBases(); // Populate the dropdown
  });
}

if (closeSpan && kbModal) {
  // Close modal when "X" is clicked
  closeSpan.addEventListener('click', () => {
    kbModal.style.display = 'none';
  });
}

// Close modal if user clicks outside the modal-content
window.onclick = function(event) {
  if (event.target === kbModal) {
    kbModal.style.display = 'none';
  }
};

// -------------------- Populate KB Dropdown --------------------
function loadKnowledgeBases() {
  fetch('/kb_list')
    .then(response => response.json())
    .then(data => {
      const kbFolderSelect = document.getElementById('kb_folder');
      if (kbFolderSelect) {
        kbFolderSelect.innerHTML = '';
        data.forEach(folderName => {
          const option = document.createElement('option');
          option.value = folderName;
          option.textContent = folderName;
          kbFolderSelect.appendChild(option);
        });
      }
    })
    .catch(err => console.error('Error loading KB list:', err));
}

// -------------------- Drag & Drop Upload Logic --------------------
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => e.preventDefault());

const uploadWindow = document.getElementById('uploadWindow');
const fileInput = document.getElementById('fileInput');
const previewBox = document.getElementById('previewBox');

if (uploadWindow && fileInput && previewBox) {
  // Clicking on uploadWindow -> open file dialog
  uploadWindow.addEventListener('click', () => {
    fileInput.click();
  });

  function updatePreview(fileList) {
    if (fileList && fileList.length > 0) {
      const file = fileList[0];
      previewBox.innerHTML = `
        <p><strong>File Name:</strong> ${file.name}</p>
        <p><strong>File Size:</strong> ${file.size} bytes</p>
      `;
    } else {
      previewBox.innerHTML = `<p>No file selected.</p>`;
    }
  }

  // Manual selection
  fileInput.addEventListener('change', e => {
    updatePreview(e.target.files);
  });

  // Drag events
  uploadWindow.addEventListener('dragover', e => {
    e.preventDefault();
    e.stopPropagation();
    uploadWindow.classList.add('dragover');
  });

  uploadWindow.addEventListener('dragleave', e => {
    e.preventDefault();
    e.stopPropagation();
    uploadWindow.classList.remove('dragover');
  });

  uploadWindow.addEventListener('drop', e => {
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
