/* ── 요소 참조 ── */
const dropZone    = document.getElementById('drop-zone');
const fileInput   = document.getElementById('file-input');
const dropContent = document.getElementById('drop-content');
const previewWrap = document.getElementById('preview-wrap');
const previewImg  = document.getElementById('preview-img');
const btnChange   = document.getElementById('btn-change');
const btnAnalyze  = document.getElementById('btn-analyze');
const loading     = document.getElementById('loading');
const errorBox    = document.getElementById('error-box');
const resultSection = document.getElementById('result-section');
const tagList     = document.getElementById('tag-list');
const addInput    = document.getElementById('add-input');
const btnAdd      = document.getElementById('btn-add');
const btnNext     = document.getElementById('btn-next');

let imageDataUrl = null;   // base64 data URI
let ingredients  = [];     // 확정된 재료 배열

/* ── 파일 선택 ── */
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

dropZone.addEventListener('click', (e) => {
  if (e.target === btnChange || e.target.closest('#preview-wrap')) return;
  fileInput.click();
});

dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') fileInput.click();
});

btnChange.addEventListener('click', (e) => {
  e.stopPropagation();
  fileInput.click();
});

/* ── 드래그&드롭 ── */
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

['dragleave', 'dragend'].forEach((ev) =>
  dropZone.addEventListener(ev, () => dropZone.classList.remove('dragover'))
);

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) loadFile(file);
});

/* ── 파일 로드 ── */
function loadFile(file) {
  const allowed = ['image/jpeg', 'image/png', 'image/webp'];
  if (!allowed.includes(file.type)) {
    showError('JPEG, PNG, WEBP 형식의 이미지만 업로드할 수 있습니다.');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError('파일 크기는 10MB 이하여야 합니다.');
    return;
  }

  hideError();
  const reader = new FileReader();
  reader.onload = (e) => {
    imageDataUrl = e.target.result;
    previewImg.src = imageDataUrl;
    dropContent.classList.add('hidden');
    previewWrap.classList.remove('hidden');
    btnAnalyze.disabled = false;
    // 이전 결과 초기화
    resultSection.classList.add('hidden');
    ingredients = [];
  };
  reader.readAsDataURL(file);
}

/* ── 재료 분석 ── */
btnAnalyze.addEventListener('click', async () => {
  if (!imageDataUrl) return;

  hideError();
  setLoading(true);
  btnAnalyze.disabled = true;

  try {
    const resp = await fetch('/api/recognize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageDataUrl }),
    });

    const data = await resp.json();

    if (!resp.ok) {
      throw new Error(data.error || '서버 오류가 발생했습니다.');
    }

    ingredients = data.ingredients || [];
    if (ingredients.length === 0) {
      showError('재료를 인식하지 못했습니다. 더 선명한 냉장고 사진을 사용해 주세요.');
    } else {
      renderTags();
      resultSection.classList.remove('hidden');
    }
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
    btnAnalyze.disabled = false;
  }
});

/* ── 태그 렌더링 ── */
function renderTags() {
  tagList.innerHTML = '';
  ingredients.forEach((item, idx) => {
    const tag = document.createElement('span');
    tag.className = 'tag';
    tag.innerHTML = `
      ${escapeHtml(item)}
      <button class="tag-remove" data-idx="${idx}" aria-label="${item} 삭제">×</button>
    `;
    tagList.appendChild(tag);
  });

  tagList.querySelectorAll('.tag-remove').forEach((btn) => {
    btn.addEventListener('click', () => {
      const idx = Number(btn.dataset.idx);
      ingredients.splice(idx, 1);
      renderTags();
    });
  });
}

/* ── 재료 추가 ── */
function addIngredient() {
  const val = addInput.value.trim();
  if (!val) return;
  if (ingredients.includes(val)) {
    addInput.value = '';
    return;
  }
  ingredients.push(val);
  addInput.value = '';
  renderTags();
}

btnAdd.addEventListener('click', addIngredient);
addInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') addIngredient();
});

/* ── Step 2 이동 ── */
btnNext.addEventListener('click', () => {
  if (ingredients.length === 0) {
    showError('재료가 없습니다. 재료를 추가해 주세요.');
    return;
  }
  // Step 2 구현 전: sessionStorage에 저장 후 step2.html로 이동
  sessionStorage.setItem('ingredients', JSON.stringify(ingredients));
  window.location.href = 'step2.html';
});

/* ── 유틸 ── */
function setLoading(on) {
  loading.classList.toggle('hidden', !on);
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove('hidden');
}

function hideError() {
  errorBox.classList.add('hidden');
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
