/* ── 요소 참조 ── */
const ingredientSummary = document.getElementById('ingredient-summary');
const loading    = document.getElementById('loading');
const errorBox   = document.getElementById('error-box');
const cardGrid   = document.getElementById('card-grid');
const actionBar  = document.getElementById('action-bar');
const btnBack    = document.getElementById('btn-back');
const btnRetry   = document.getElementById('btn-retry');
const modalOverlay = document.getElementById('modal-overlay');
const modalClose   = document.getElementById('modal-close');
const modalContent = document.getElementById('modal-content');

/* ── 재료 목록 로드 ── */
let ingredients = [];
try {
  ingredients = JSON.parse(sessionStorage.getItem('ingredients') || '[]');
} catch {
  ingredients = [];
}

if (ingredients.length === 0) {
  showError('재료 정보가 없습니다. Step 1로 돌아가 재료를 인식해 주세요.');
  loading.classList.add('hidden');
} else if (ingredients.length < 2) {
  showError('레시피 생성을 위해 재료가 2개 이상 필요합니다. 재료를 추가해 주세요.');
  loading.classList.add('hidden');
} else {
  ingredientSummary.textContent = '재료: ' + ingredients.join(', ');
  fetchRecipes();
}

/* ── 레시피 요청 ── */
async function fetchRecipes() {
  showLoading(true);
  hideError();
  cardGrid.classList.add('hidden');
  actionBar.classList.add('hidden');

  // Step 3 프로필 조건 읽기 (있을 경우)
  let conditions = '';
  try {
    const profile = JSON.parse(localStorage.getItem('userProfile') || '{}');
    const parts = [];
    if (profile.dietRestrictions?.length) parts.push(profile.dietRestrictions.join(', '));
    if (profile.avoidIngredients?.length) parts.push(`${profile.avoidIngredients.join(', ')}은(는) 사용하지 마세요`);
    conditions = parts.join('. ');
  } catch { /* 프로필 없음 — 무시 */ }

  try {
    const resp = await fetch('/api/recipe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ingredients, conditions }),
    });
    const data = await resp.json();

    if (!resp.ok) throw new Error(data.error || '서버 오류가 발생했습니다.');
    if (!data.recipes || data.recipes.length === 0) throw new Error('레시피를 생성하지 못했습니다. 다시 시도해 주세요.');

    renderCards(data.recipes);
  } catch (err) {
    showError(err.message);
  } finally {
    showLoading(false);
  }
}

/* ── 카드 렌더링 ── */
function renderCards(recipes) {
  cardGrid.innerHTML = '';
  recipes.forEach((recipe, idx) => {
    const card = document.createElement('div');
    card.className = 'recipe-card';
    card.innerHTML = `
      <div class="card-header">
        <h3 class="card-title">${escapeHtml(recipe.name || '레시피 ' + (idx + 1))}</h3>
        <p class="card-desc">${escapeHtml(recipe.description || '')}</p>
      </div>
      <div class="card-badges">
        <span class="badge badge-time">⏱ ${escapeHtml(recipe.cooking_time || '?')}</span>
        <span class="badge badge-diff ${diffClass(recipe.difficulty)}">${diffEmoji(recipe.difficulty)} ${escapeHtml(recipe.difficulty || '?')}</span>
      </div>
      <ul class="card-ingredients">
        ${(recipe.ingredients || []).slice(0, 4).map(i => `<li>${escapeHtml(i)}</li>`).join('')}
        ${(recipe.ingredients || []).length > 4 ? `<li class="more">+${recipe.ingredients.length - 4}개 더</li>` : ''}
      </ul>
      <div class="card-actions">
        <button class="btn-outline btn-detail" data-idx="${idx}">상세 보기</button>
        <button class="btn-save" data-idx="${idx}">저장하기 ♡</button>
      </div>
    `;
    cardGrid.appendChild(card);
  });

  // 상세 보기
  cardGrid.querySelectorAll('.btn-detail').forEach(btn => {
    btn.addEventListener('click', () => openModal(recipes[Number(btn.dataset.idx)]));
  });

  // 저장하기
  cardGrid.querySelectorAll('.btn-save').forEach(btn => {
    btn.addEventListener('click', () => saveRecipe(recipes[Number(btn.dataset.idx)], btn));
  });

  cardGrid.classList.remove('hidden');
  actionBar.classList.remove('hidden');
}

/* ── 모달 ── */
function openModal(recipe) {
  modalContent.innerHTML = `
    <h2 class="modal-title">${escapeHtml(recipe.name || '')}</h2>
    <p class="modal-desc">${escapeHtml(recipe.description || '')}</p>
    <div class="card-badges" style="margin: 12px 0">
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

/* ── 저장 ── */
function saveRecipe(recipe, btn) {
  const profile = JSON.parse(localStorage.getItem('userProfile') || 'null');
  if (!profile) {
    // 프로필 없음 → Step 3으로 이동하며 저장할 레시피 전달
    sessionStorage.setItem('pendingRecipe', JSON.stringify({ recipe, ingredients }));
    window.location.href = 'step3.html';
    return;
  }

  // 프로필 있음 → 바로 저장
  const saved = JSON.parse(localStorage.getItem('savedRecipes') || '[]');
  const isDuplicate = saved.some(r => r.recipe.name === recipe.name);
  if (isDuplicate) {
    btn.textContent = '이미 저장됨 ✓';
    btn.disabled = true;
    return;
  }
  saved.push({ recipe, ingredients, savedAt: new Date().toISOString(), model: 'nvidia/nemotron-3-super-120b-a12b:free' });
  localStorage.setItem('savedRecipes', JSON.stringify(saved));
  btn.textContent = '저장됨 ✓';
  btn.disabled = true;
}

/* ── 네비게이션 ── */
btnBack.addEventListener('click', () => { window.location.href = '/'; });
btnRetry.addEventListener('click', fetchRecipes);

/* ── 유틸 ── */
function showLoading(on) { loading.classList.toggle('hidden', !on); }
function showError(msg) { errorBox.textContent = msg; errorBox.classList.remove('hidden'); }
function hideError() { errorBox.classList.add('hidden'); }
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
