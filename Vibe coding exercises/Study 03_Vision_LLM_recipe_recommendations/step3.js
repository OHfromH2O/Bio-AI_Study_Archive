/* ════════════════════════════════════════
   Step 3 — 사용자 프로필 & 레시피 보관함
════════════════════════════════════════ */

const DEFAULT_AVATAR = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 80 80'%3E%3Ccircle cx='40' cy='40' r='40' fill='%23e2e8f0'/%3E%3Ccircle cx='40' cy='30' r='14' fill='%23a0aec0'/%3E%3Cellipse cx='40' cy='72' rx='22' ry='16' fill='%23a0aec0'/%3E%3C/svg%3E";

/* ── localStorage 키 ── */
const PROFILE_KEY = 'userProfile';
const RECIPES_KEY = 'savedRecipes';

/* ── 패널 참조 ── */
const setupPanel = document.getElementById('setup-panel');
const mainPanel  = document.getElementById('main-panel');

/* ────────────────────────────────────────
   초기화: 프로필 유무에 따라 패널 결정
──────────────────────────────────────── */
function init() {
  const profile = loadProfile();
  if (profile) {
    showMainPanel(profile);
    // pendingRecipe 처리 (Step 2에서 넘어온 경우)
    const pending = sessionStorage.getItem('pendingRecipe');
    if (pending) {
      try {
        const { recipe, ingredients } = JSON.parse(pending);
        persistRecipe(recipe, ingredients);
        sessionStorage.removeItem('pendingRecipe');
      } catch { /* 무시 */ }
    }
    renderBookshelf();
  } else {
    showSetupPanel();
  }
  checkStorageWarning();
}

/* ════════════════════════════════════════
   프로필 생성 패널 (Setup)
════════════════════════════════════════ */
let setupAvatarB64 = '';
let setupAvoidList = [];

function showSetupPanel() {
  setupPanel.classList.remove('hidden');
  mainPanel.classList.add('hidden');
  renderAvoidTags('setup-avoid-tags', setupAvoidList, () => {});
  document.getElementById('setup-avatar-preview').src = DEFAULT_AVATAR;
}

/* 아바타 업로드 */
setupAvatarInput('setup-avatar-input', 'setup-avatar-preview', (b64) => { setupAvatarB64 = b64; });

/* 기피 재료 추가 */
addTagBehavior(
  'setup-avoid-input', 'setup-avoid-add',
  () => setupAvoidList,
  (list) => { setupAvoidList = list; renderAvoidTags('setup-avoid-tags', setupAvoidList, () => {}); }
);

/* 저장 */
document.getElementById('setup-save').addEventListener('click', () => {
  const nickname = document.getElementById('setup-nickname').value.trim();
  const errEl = document.getElementById('setup-error');
  if (!nickname) {
    errEl.textContent = '닉네임을 입력해 주세요.';
    errEl.classList.remove('hidden');
    return;
  }
  errEl.classList.add('hidden');

  const profile = {
    nickname,
    avatar: setupAvatarB64 || DEFAULT_AVATAR,
    dietRestrictions: getCheckedDiets('#setup-panel'),
    avoidIngredients: [...setupAvoidList],
    createdAt: new Date().toISOString(),
  };
  saveProfile(profile);

  // pendingRecipe 저장
  const pending = sessionStorage.getItem('pendingRecipe');
  if (pending) {
    try {
      const { recipe, ingredients } = JSON.parse(pending);
      persistRecipe(recipe, ingredients);
      sessionStorage.removeItem('pendingRecipe');
    } catch { /* 무시 */ }
  }

  showMainPanel(profile);
  renderBookshelf();
});

/* ════════════════════════════════════════
   메인 패널
════════════════════════════════════════ */
function showMainPanel(profile) {
  setupPanel.classList.add('hidden');
  mainPanel.classList.remove('hidden');

  document.getElementById('main-avatar').src    = profile.avatar || DEFAULT_AVATAR;
  document.getElementById('main-nickname').textContent = profile.nickname;

  // 식이 제한 뱃지
  const dietEl = document.getElementById('main-diets');
  dietEl.innerHTML = (profile.dietRestrictions || [])
    .map(d => `<span class="badge badge-diet">${escapeHtml(d)}</span>`).join('');

  // 기피 재료 태그 (삭제 없이 표시만)
  const avoidEl = document.getElementById('main-avoids');
  avoidEl.innerHTML = (profile.avoidIngredients || []).length
    ? (profile.avoidIngredients).map(a => `<span class="tag tag-avoid">${escapeHtml(a)}</span>`).join('')
    : '<span class="hint-label">없음</span>';
}

/* ════════════════════════════════════════
   프로필 편집 패널
════════════════════════════════════════ */
let editAvoidList = [];
let editAvatarB64 = '';

document.getElementById('btn-edit-profile').addEventListener('click', () => {
  const profile = loadProfile();
  editAvatarB64 = profile.avatar || DEFAULT_AVATAR;
  editAvoidList = [...(profile.avoidIngredients || [])];

  document.getElementById('edit-avatar-preview').src = editAvatarB64;
  document.getElementById('edit-nickname').value     = profile.nickname;

  // 체크박스 복원
  document.querySelectorAll('#edit-diet-options input[type=checkbox]').forEach(cb => {
    cb.checked = (profile.dietRestrictions || []).includes(cb.value);
  });

  renderAvoidTags('edit-avoid-tags', editAvoidList, () => {
    renderAvoidTags('edit-avoid-tags', editAvoidList, () => {});
  });

  document.getElementById('edit-panel').classList.remove('hidden');
});

setupAvatarInput('edit-avatar-input', 'edit-avatar-preview', (b64) => { editAvatarB64 = b64; });

addTagBehavior(
  'edit-avoid-input', 'edit-avoid-add',
  () => editAvoidList,
  (list) => { editAvoidList = list; renderAvoidTags('edit-avoid-tags', editAvoidList, () => {}); }
);

document.getElementById('edit-cancel').addEventListener('click', () => {
  document.getElementById('edit-panel').classList.add('hidden');
});

document.getElementById('edit-save').addEventListener('click', () => {
  const nickname = document.getElementById('edit-nickname').value.trim();
  if (!nickname) return;

  const profile = {
    ...loadProfile(),
    nickname,
    avatar: editAvatarB64 || DEFAULT_AVATAR,
    dietRestrictions: getCheckedDiets('#edit-panel'),
    avoidIngredients: [...editAvoidList],
  };
  saveProfile(profile);
  showMainPanel(profile);
  document.getElementById('edit-panel').classList.add('hidden');
});

/* 데이터 초기화 */
document.getElementById('btn-reset').addEventListener('click', () => {
  if (!confirm('프로필과 저장된 레시피를 모두 삭제합니다. 계속하시겠습니까?')) return;
  localStorage.removeItem(PROFILE_KEY);
  localStorage.removeItem(RECIPES_KEY);
  location.reload();
});

/* ════════════════════════════════════════
   레시피 보관함
════════════════════════════════════════ */
function renderBookshelf() {
  const saved  = loadRecipes();
  const sort   = document.getElementById('sort-select').value;
  const title  = document.getElementById('bookshelf-title');
  const listEl = document.getElementById('recipe-list');
  const empty  = document.getElementById('empty-state');

  title.textContent = `📋 저장된 레시피 (${saved.length}개)`;

  if (saved.length === 0) {
    empty.classList.remove('hidden');
    listEl.innerHTML = '';
    return;
  }
  empty.classList.add('hidden');

  const sorted = [...saved].sort((a, b) => {
    if (sort === 'name') return (a.recipe.name || '').localeCompare(b.recipe.name || '');
    return new Date(b.savedAt) - new Date(a.savedAt);
  });

  listEl.innerHTML = '';
  sorted.forEach((item, idx) => {
    const originalIdx = saved.indexOf(item);
    const card = document.createElement('div');
    card.className = 'saved-card';
    card.innerHTML = `
      <div class="saved-card-top">
        <div>
          <h4 class="saved-name">${escapeHtml(item.recipe.name || '')}</h4>
          <p class="saved-meta">
            ${formatDate(item.savedAt)} &nbsp;|&nbsp;
            <span class="badge badge-time">⏱ ${escapeHtml(item.recipe.cooking_time || '?')}</span>
            <span class="badge badge-diff ${diffClass(item.recipe.difficulty)}">${diffEmoji(item.recipe.difficulty)} ${escapeHtml(item.recipe.difficulty || '?')}</span>
          </p>
          <p class="saved-source">재료: ${(item.ingredients || []).join(', ')}</p>
        </div>
        <div class="saved-actions">
          <button class="btn-outline btn-sm btn-detail-s" data-idx="${originalIdx}">상세</button>
          <button class="btn-danger-sm btn-del" data-idx="${originalIdx}">삭제</button>
        </div>
      </div>
      <div class="memo-wrap">
        <textarea class="memo-input" placeholder="메모를 입력하세요…" data-idx="${originalIdx}">${escapeHtml(item.memo || '')}</textarea>
        <button class="btn-secondary btn-sm btn-memo-save" data-idx="${originalIdx}">저장</button>
      </div>
    `;
    listEl.appendChild(card);
  });

  /* 상세 보기 */
  listEl.querySelectorAll('.btn-detail-s').forEach(btn => {
    btn.addEventListener('click', () => openModal(saved[Number(btn.dataset.idx)].recipe));
  });

  /* 삭제 */
  listEl.querySelectorAll('.btn-del').forEach(btn => {
    btn.addEventListener('click', () => {
      if (!confirm('이 레시피를 삭제하시겠습니까?')) return;
      const all = loadRecipes();
      all.splice(Number(btn.dataset.idx), 1);
      localStorage.setItem(RECIPES_KEY, JSON.stringify(all));
      renderBookshelf();
      checkStorageWarning();
    });
  });

  /* 메모 저장 */
  listEl.querySelectorAll('.btn-memo-save').forEach(btn => {
    btn.addEventListener('click', () => {
      const idx      = Number(btn.dataset.idx);
      const textarea = listEl.querySelector(`.memo-input[data-idx="${idx}"]`);
      const all      = loadRecipes();
      all[idx].memo  = textarea.value;
      localStorage.setItem(RECIPES_KEY, JSON.stringify(all));
      btn.textContent = '저장됨 ✓';
      setTimeout(() => { btn.textContent = '저장'; }, 1500);
    });
  });
}

document.getElementById('sort-select').addEventListener('change', renderBookshelf);

/* ════════════════════════════════════════
   모달 (Step 2와 동일 레이아웃)
════════════════════════════════════════ */
const modalOverlay = document.getElementById('modal-overlay');
const modalClose   = document.getElementById('modal-close');
const modalBody    = document.getElementById('modal-body');

function openModal(recipe) {
  modalBody.innerHTML = `
    <h2 class="modal-title">${escapeHtml(recipe.name || '')}</h2>
    <p class="modal-desc">${escapeHtml(recipe.description || '')}</p>
    <div class="card-badges" style="margin:12px 0">
      <span class="badge badge-time">⏱ ${escapeHtml(recipe.cooking_time || '?')}</span>
      <span class="badge badge-diff ${diffClass(recipe.difficulty)}">${diffEmoji(recipe.difficulty)} ${escapeHtml(recipe.difficulty || '?')}</span>
    </div>
    <h4>재료</h4>
    <ul class="modal-list">
      ${(recipe.ingredients || []).map(i => `<li>${escapeHtml(i)}</li>`).join('')}
    </ul>
    <h4>조리 방법</h4>
    <ol class="modal-steps">
      ${(recipe.steps || []).map(s => `<li>${escapeHtml(s)}</li>`).join('')}
    </ol>
  `;
  modalOverlay.classList.remove('hidden');
  document.body.style.overflow = 'hidden';
}

function closeModal() {
  modalOverlay.classList.add('hidden');
  document.body.style.overflow = '';
}
modalClose.addEventListener('click', closeModal);
modalOverlay.addEventListener('click', e => { if (e.target === modalOverlay) closeModal(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

/* ════════════════════════════════════════
   localStorage 헬퍼
════════════════════════════════════════ */
function loadProfile()  { try { return JSON.parse(localStorage.getItem(PROFILE_KEY)); } catch { return null; } }
function saveProfile(p) { localStorage.setItem(PROFILE_KEY, JSON.stringify(p)); }
function loadRecipes()  { try { return JSON.parse(localStorage.getItem(RECIPES_KEY)) || []; } catch { return []; } }

function persistRecipe(recipe, ingredients) {
  const all = loadRecipes();
  if (all.some(r => r.recipe.name === recipe.name)) return; // 중복 방지
  all.push({
    recipe,
    ingredients,
    savedAt: new Date().toISOString(),
    model: 'nvidia/nemotron-3-super-120b-a12b:free',
    memo: '',
  });
  localStorage.setItem(RECIPES_KEY, JSON.stringify(all));
}

function checkStorageWarning() {
  try {
    const used = JSON.stringify(localStorage).length * 2; // bytes (UTF-16)
    const LIMIT = 5 * 1024 * 1024;
    document.getElementById('storage-warning').classList.toggle('hidden', used < LIMIT * 0.85);
  } catch { /* 무시 */ }
}

/* ════════════════════════════════════════
   공통 유틸
════════════════════════════════════════ */
function setupAvatarInput(inputId, previewId, onLoad) {
  document.getElementById(inputId).addEventListener('change', function () {
    if (!this.files[0]) return;
    const reader = new FileReader();
    reader.onload = e => {
      document.getElementById(previewId).src = e.target.result;
      onLoad(e.target.result);
    };
    reader.readAsDataURL(this.files[0]);
  });
}

function addTagBehavior(inputId, btnId, getList, setList) {
  const addFn = () => {
    const input = document.getElementById(inputId);
    const val   = input.value.trim();
    if (!val) return;
    const list = getList();
    if (!list.includes(val)) setList([...list, val]);
    input.value = '';
  };
  document.getElementById(btnId).addEventListener('click', addFn);
  document.getElementById(inputId).addEventListener('keydown', e => { if (e.key === 'Enter') addFn(); });
}

function renderAvoidTags(containerId, list, onChange) {
  const el = document.getElementById(containerId);
  el.innerHTML = list.map((item, i) => `
    <span class="tag">
      ${escapeHtml(item)}
      <button class="tag-remove" data-i="${i}" aria-label="${item} 삭제">×</button>
    </span>
  `).join('');
  el.querySelectorAll('.tag-remove').forEach(btn => {
    btn.addEventListener('click', () => {
      list.splice(Number(btn.dataset.i), 1);
      renderAvoidTags(containerId, list, onChange);
      onChange(list);
    });
  });
}

function getCheckedDiets(scopeSelector) {
  return [...document.querySelectorAll(`${scopeSelector} .diet-chip input:checked`)]
    .map(cb => cb.value);
}

function formatDate(iso) {
  return iso ? iso.slice(0, 10).replace(/-/g, '.') : '';
}

function escapeHtml(str) {
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function diffClass(d) {
  if (!d) return '';
  if (d.includes('쉬')) return 'diff-easy';
  if (d.includes('어')) return 'diff-hard';
  return 'diff-mid';
}

function diffEmoji(d) {
  if (!d) return '⭐';
  if (d.includes('쉬')) return '🟢';
  if (d.includes('어')) return '🔴';
  return '🟡';
}

/* ── 시작 ── */
init();
