/* ══════════════════════════════════════════════════════════
   My Tasks – script.js
   최종 완성본: 다크모드 / 카테고리 / 검색 / 정렬 / 드래그앤드롭
               / 인라인 수정 / 대시보드 / 내보내기·가져오기
               / 실행취소(Undo) / 접근성 / 키보드 단축키
══════════════════════════════════════════════════════════ */

/* ──────────────────────────────────────────────────────────
   1. 상수 정의
────────────────────────────────────────────────────────── */

/** localStorage 키 */
const STORAGE_KEY = 'my_tasks';
const FILTER_KEY  = 'my_tasks_filter';
const THEME_KEY   = 'my_tasks_theme';
const SORT_KEY    = 'my_tasks_sort';

/** 카테고리 메타 정보 */
const CATEGORIES = {
  work:     { label: '업무', emoji: '💼' },
  personal: { label: '개인', emoji: '🌿' },
  study:    { label: '공부', emoji: '📚' },
};

/** 정렬 모드 */
const SORT_MODES = {
  manual:   '직접 정렬',
  newest:   '최신순',
  oldest:   '오래된순',
  category: '카테고리순',
};

/** 완료율에 따른 응원 메시지 */
const ENCOURAGEMENTS = [
  { min: 0,   max: 0,   msg: '오늘도 화이팅! 첫 번째 할 일을 완료해봐요 🚀' },
  { min: 1,   max: 25,  msg: '좋은 시작이에요! 계속 나아가요 💪' },
  { min: 26,  max: 50,  msg: '절반을 향해 달리는 중이에요! 잘 하고 있어요 👍' },
  { min: 51,  max: 75,  msg: '절반을 넘었어요! 이제 마무리만 남았어요 ✨' },
  { min: 76,  max: 99,  msg: '거의 다 왔어요! 마지막 스퍼트! 🔥' },
  { min: 100, max: 100, msg: '모든 할 일 완료! 오늘 정말 대단해요! 🎉' },
];

/** 오늘의 격언 목록 (페이지 로드 시 무작위 선택) */
const QUOTES = [
  { text: '시작이 반이다.',                    author: '아리스토텔레스' },
  { text: '천 리 길도 한 걸음부터.',            author: '노자' },
  { text: '할 수 있다고 믿는 사람이 할 수 있다.', author: '나폴레옹' },
  { text: '오늘 할 일을 내일로 미루지 마라.',    author: '벤자민 프랭클린' },
  { text: '가장 중요한 일에 집중하라.',          author: '괴테' },
  { text: '작은 진전도 진전이다.',               author: '' },
  { text: '당신이 할 수 있는 일을 하라.',        author: '테오도어 루스벨트' },
  { text: '성공은 준비와 기회가 만나는 곳에 있다.', author: '세네카' },
  { text: '집중하면 불가능은 없다.',             author: '' },
  { text: '하루하루가 새로운 기회다.',           author: '' },
];

/* ──────────────────────────────────────────────────────────
   2. 애플리케이션 상태
────────────────────────────────────────────────────────── */
let tasks        = loadTasks();           // 할 일 배열
let activeFilter = localStorage.getItem(FILTER_KEY) || 'all';
let sortBy       = localStorage.getItem(SORT_KEY)   || 'manual';
let searchQuery  = '';

/** 마지막으로 삭제된 항목 (Undo용) */
let undoStack = null; // { task, index }

/** 드래그 중인 항목 id */
let dragSrcId = null;

/* ──────────────────────────────────────────────────────────
   3. DOM 참조
────────────────────────────────────────────────────────── */
const taskInput         = document.getElementById('taskInput');
const addBtn            = document.getElementById('addBtn');
const categorySelect    = document.getElementById('categorySelect');
const taskList          = document.getElementById('taskList');
const emptyMessage      = document.getElementById('emptyMessage');
const filterBtns        = document.querySelectorAll('.filter-btn');
const searchInput       = document.getElementById('searchInput');
const searchClear       = document.getElementById('searchClear');
const searchStatus      = document.getElementById('searchStatus');
const remainingBadge    = document.getElementById('remainingBadge');
const clearCompletedBtn = document.getElementById('clearCompletedBtn');
const themeToggle       = document.getElementById('themeToggle');
const themeIcon         = document.getElementById('themeIcon');
const sortSelect        = document.getElementById('sortSelect');
const exportBtn         = document.getElementById('exportBtn');
const importBtn         = document.getElementById('importBtn');
const importFile        = document.getElementById('importFile');
const quoteText         = document.getElementById('quoteText');
const encouragement     = document.getElementById('encouragement');
const progressTrack     = document.getElementById('progressTrack');
const progressBar       = document.getElementById('progressBar');
const dashSummary       = document.getElementById('dashSummary');
const dashTodayCount    = document.getElementById('dashTodayCount');
const dashWorkCount     = document.getElementById('dashWorkCount');
const barWork           = document.getElementById('barWork');
const dashPersonalCount = document.getElementById('dashPersonalCount');
const barPersonal       = document.getElementById('barPersonal');
const dashStudyCount    = document.getElementById('dashStudyCount');
const barStudy          = document.getElementById('barStudy');
const ariaAnnounce      = document.getElementById('ariaAnnounce');

/* ──────────────────────────────────────────────────────────
   4. 유틸리티 함수
────────────────────────────────────────────────────────── */

/**
 * 디바운스: 연속 호출 시 마지막 호출 후 delay ms 뒤에만 실행.
 * 검색 입력의 렌더링 빈도를 제한하는 데 사용한다.
 */
function debounce(fn, delay) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

/**
 * 문자열을 최대 len 글자로 잘라 "..." 을 붙인다.
 * 토스트 메시지에서 긴 텍스트를 요약할 때 사용.
 */
function truncate(str, len = 22) {
  return str.length > len ? str.slice(0, len) + '…' : str;
}

/**
 * ISO 날짜 문자열을 사람이 읽기 쉬운 상대 시간으로 변환.
 * 예: "방금 전", "3분 전", "2시간 전", "5일 전"
 */
function timeAgo(iso) {
  const diff = (Date.now() - new Date(iso).getTime()) / 1000; // 초 단위
  if (diff < 60)    return '방금 전';
  if (diff < 3600)  return `${Math.floor(diff / 60)}분 전`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}시간 전`;
  return `${Math.floor(diff / 86400)}일 전`;
}

/**
 * 검색어 하이라이트: 텍스트 내 검색어를 <mark>로 감싼다.
 * XSS 방어를 위해 HTML 특수문자를 먼저 이스케이프한다.
 */
function highlight(text, query) {
  // 1) HTML 특수문자 이스케이프 (XSS 방어)
  const safe = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // 2) 정규식 특수문자 이스케이프 후 검색어 치환
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return safe.replace(new RegExp(`(${escaped})`, 'gi'), '<mark>$1</mark>');
}

/**
 * ARIA 공지: 스크린 리더에게 변경 내용을 읽어준다.
 * requestAnimationFrame으로 DOM 업데이트 후 텍스트를 설정해
 * 같은 메시지도 재공지되도록 한다.
 */
function announce(msg) {
  ariaAnnounce.textContent = '';
  requestAnimationFrame(() => { ariaAnnounce.textContent = msg; });
}

/* ──────────────────────────────────────────────────────────
   5. 토스트 알림
────────────────────────────────────────────────────────── */

/**
 * 화면 하단 중앙에 토스트 알림을 표시한다.
 * @param {string}   message  - 표시할 텍스트
 * @param {object|null} action - { label, fn } 형태의 액션 버튼 (선택적)
 * @param {number}   duration - 자동 닫힘 시간 ms (기본 4000)
 */
function showToast(message, action = null, duration = 4000) {
  const container = document.getElementById('toastContainer');

  // 기존 토스트 즉시 제거 (중복 방지)
  container.innerHTML = '';

  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.setAttribute('role', 'alert');

  // 메시지 텍스트
  const msgEl = document.createElement('span');
  msgEl.style.flex = '1';
  msgEl.textContent = message;
  toast.appendChild(msgEl);

  // 액션 버튼 (예: "실행 취소")
  if (action) {
    const actionBtn = document.createElement('button');
    actionBtn.className = 'toast-action';
    actionBtn.textContent = action.label;
    actionBtn.addEventListener('click', () => {
      action.fn();
      dismissToast(toast);
    });
    toast.appendChild(actionBtn);
  }

  // 닫기 버튼
  const closeBtn = document.createElement('button');
  closeBtn.className = 'toast-close';
  closeBtn.textContent = '✕';
  closeBtn.setAttribute('aria-label', '알림 닫기');
  closeBtn.addEventListener('click', () => dismissToast(toast));
  toast.appendChild(closeBtn);

  container.appendChild(toast);

  // 자동 닫힘
  const timer = setTimeout(() => dismissToast(toast), duration);

  // 액션 버튼 클릭 시 타이머 취소 (클릭 후 즉시 닫힘 처리됨)
  toast._timer = timer;
}

/** 토스트를 페이드아웃 후 제거한다. */
function dismissToast(toast) {
  clearTimeout(toast._timer);
  toast.classList.add('toast-out');
  toast.addEventListener('animationend', () => toast.remove(), { once: true });
}

/* ──────────────────────────────────────────────────────────
   6. 다크모드
────────────────────────────────────────────────────────── */

/** 저장된 테마를 불러와 적용한다. */
function initTheme() {
  setTheme(localStorage.getItem(THEME_KEY) || 'light');
}

/** 토글: 현재 테마의 반대로 전환한다. */
function toggleTheme() {
  setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
}

/**
 * 테마를 적용하고 localStorage에 저장한다.
 * – html[data-theme] 변경 → CSS 변수 일괄 전환
 * – 토글 체크박스, 아이콘, ARIA 속성 동기화
 */
function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const isDark = theme === 'dark';
  themeToggle.checked = isDark;
  themeToggle.setAttribute('aria-checked', String(isDark));
  themeIcon.textContent = isDark ? '☀️' : '🌙';
  localStorage.setItem(THEME_KEY, theme);
}

/* ──────────────────────────────────────────────────────────
   7. 오늘의 격언
────────────────────────────────────────────────────────── */

/** 격언 배열에서 무작위로 하나 골라 표시한다. */
function initQuote() {
  const q = QUOTES[Math.floor(Math.random() * QUOTES.length)];
  quoteText.textContent = q.author
    ? `"${q.text}" — ${q.author}`
    : `"${q.text}"`;
}

/* ──────────────────────────────────────────────────────────
   8. 할 일 추가
────────────────────────────────────────────────────────── */

function handleAdd() {
  const text = taskInput.value.trim();

  // 빈 입력: 흔들림 피드백
  if (!text) {
    taskInput.classList.add('shake');
    taskInput.addEventListener('animationend', () => taskInput.classList.remove('shake'), { once: true });
    taskInput.focus();
    return;
  }

  // 중복 확인: 대소문자 무관하게 동일한 텍스트가 있는지 검사
  const duplicate = tasks.find(
    (t) => t.text.trim().toLowerCase() === text.toLowerCase()
  );
  if (duplicate) {
    const proceed = window.confirm(
      `"${truncate(text, 30)}" 항목이 이미 있습니다.\n그래도 추가할까요?`
    );
    if (!proceed) {
      taskInput.focus();
      return;
    }
  }

  const task = {
    id:        Date.now().toString(),
    text,
    category:  categorySelect.value,
    completed: false,
    createdAt: new Date().toISOString(),
    order:     tasks.length,   // 기본 순서: 맨 뒤
  };

  tasks.push(task);
  saveTasks();
  renderAll();

  taskInput.value = '';
  taskInput.focus();
  announce(`"${truncate(text)}" 추가됨`);
}

/* ──────────────────────────────────────────────────────────
   9. 완료 토글
────────────────────────────────────────────────────────── */

function handleToggle(id) {
  const task = tasks.find((t) => t.id === id);
  if (!task) return;

  const li = document.querySelector(`[data-id="${id}"]`);
  const animClass = task.completed ? 'uncompleting' : 'completing';

  if (li) {
    // 애니메이션 종료 후 상태 변경 및 재렌더 (자연스러운 순서 이동)
    li.classList.add(animClass);
    li.addEventListener('animationend', () => {
      task.completed = !task.completed;
      saveTasks();
      renderAll();
      announce(task.completed ? `"${truncate(task.text)}" 완료` : `"${truncate(task.text)}" 미완료로 변경`);
    }, { once: true });
  } else {
    task.completed = !task.completed;
    saveTasks();
    renderAll();
  }
}

/* ──────────────────────────────────────────────────────────
   10. 삭제 + Undo
────────────────────────────────────────────────────────── */

function handleDelete(id) {
  const li = document.querySelector(`[data-id="${id}"]`);
  if (!li) return;

  // 삭제 전에 Undo 정보 저장
  const taskIdx    = tasks.findIndex((t) => t.id === id);
  const deletedTask = { ...tasks[taskIdx] };

  // fadeOut 애니메이션 → 실제 제거
  li.classList.add('removing');
  li.addEventListener('animationend', () => {
    tasks = tasks.filter((t) => t.id !== id);
    saveTasks();
    renderAll();

    // Undo 스택에 저장 (가장 최근 1개만 유지)
    undoStack = { task: deletedTask, index: taskIdx };

    // 실행취소 버튼이 있는 토스트 표시
    showToast(
      `"${truncate(deletedTask.text)}" 삭제됨`,
      { label: '실행 취소', fn: undoDelete },
      5000
    );
    announce(`"${truncate(deletedTask.text)}" 삭제됨. Alt+Z로 실행취소 가능`);
  }, { once: true });
}

/** 마지막으로 삭제된 항목을 원래 위치에 복원한다. */
function undoDelete() {
  if (!undoStack) return;

  const { task, index } = undoStack;
  undoStack = null;

  // 원래 인덱스(또는 배열 끝)에 삽입
  tasks.splice(Math.min(index, tasks.length), 0, task);

  // order 재정규화
  tasks.forEach((t, i) => { t.order = i; });

  saveTasks();
  renderAll();
  announce(`"${truncate(task.text)}" 복구됨`);
  showToast(`"${truncate(task.text)}" 복구됨`, null, 2500);
}

/* ──────────────────────────────────────────────────────────
   11. 완료된 항목 모두 삭제
────────────────────────────────────────────────────────── */

function handleClearCompleted() {
  const completed = tasks.filter((t) => t.completed);
  if (completed.length === 0) return;

  const ok = window.confirm(
    `완료된 항목 ${completed.length}개를 모두 삭제할까요?\n이 작업은 실행취소할 수 없습니다.`
  );
  if (!ok) return;

  tasks = tasks.filter((t) => !t.completed);
  saveTasks();
  renderAll();
  announce(`완료된 항목 ${completed.length}개 삭제됨`);
  showToast(`완료된 항목 ${completed.length}개 삭제됨`, null, 2500);
}

/* ──────────────────────────────────────────────────────────
   12. 인라인 수정
────────────────────────────────────────────────────────── */

function enterEditMode(id) {
  const task = tasks.find((t) => t.id === id);
  if (!task) return;

  const li = document.querySelector(`[data-id="${id}"]`);
  if (!li || li.dataset.editing) return; // 이미 편집 중이면 무시
  li.dataset.editing = '1';

  // 기존 body 숨기기
  const body = li.querySelector('.task-body');
  body.style.display = 'none';

  // ── 편집 UI 생성 ──
  const editRow = document.createElement('div');
  editRow.className = 'edit-row';
  editRow.setAttribute('role', 'group');
  editRow.setAttribute('aria-label', '할 일 수정');

  // 카테고리 선택
  const editCatSel = document.createElement('select');
  editCatSel.className = 'edit-category-select';
  editCatSel.setAttribute('aria-label', '카테고리 변경');
  Object.entries(CATEGORIES).forEach(([val, { label, emoji }]) => {
    const opt = document.createElement('option');
    opt.value = val;
    opt.textContent = `${emoji} ${label}`;
    if (val === task.category) opt.selected = true;
    editCatSel.appendChild(opt);
  });

  // 텍스트 입력
  const editInput = document.createElement('input');
  editInput.type = 'text';
  editInput.className = 'edit-input';
  editInput.value = task.text;
  editInput.maxLength = 200;
  editInput.setAttribute('aria-label', '할 일 내용 수정');

  // 힌트
  const hint = document.createElement('span');
  hint.className = 'edit-hint';
  hint.textContent = 'Enter 저장  ESC 취소';
  hint.setAttribute('aria-hidden', 'true');

  editRow.appendChild(editCatSel);
  editRow.appendChild(editInput);
  editRow.appendChild(hint);
  li.insertBefore(editRow, li.querySelector('.delete-btn'));

  editInput.focus();
  editInput.select();

  /** 편집 내용 저장 */
  function saveEdit() {
    const newText = editInput.value.trim();
    if (newText && newText !== task.text) {
      task.text     = newText;
      task.category = editCatSel.value;
      saveTasks();
      announce(`할 일 수정됨: "${truncate(newText)}"`);
    } else if (editCatSel.value !== task.category) {
      task.category = editCatSel.value;
      saveTasks();
    }
    renderAll();
  }

  /** 편집 취소 */
  function cancelEdit() { renderAll(); }

  // 키보드 이벤트
  editInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter')  { e.preventDefault(); saveEdit(); }
    if (e.key === 'Escape') { e.preventDefault(); cancelEdit(); }
  });

  // 편집 영역 외부 클릭 시 저장
  setTimeout(() => {
    document.addEventListener('mousedown', function onOutside(e) {
      if (!li.contains(e.target)) {
        document.removeEventListener('mousedown', onOutside);
        saveEdit();
      }
    });
  }, 0);
}

/* ──────────────────────────────────────────────────────────
   13. 드래그 앤 드롭 (수동 정렬)
────────────────────────────────────────────────────────── */

/**
 * li 요소에 HTML5 Drag and Drop 이벤트를 부착한다.
 * 마우스 Y 좌표가 항목 중심보다 위면 "앞에 삽입",
 * 아래면 "뒤에 삽입"으로 처리한다.
 */
function makeDraggable(li, taskId) {
  li.draggable = true;

  li.addEventListener('dragstart', (e) => {
    dragSrcId = taskId;
    li.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
    // 투명 드래그 이미지 방지 (기본 반투명 고스트 사용)
    e.dataTransfer.setData('text/plain', taskId);
  });

  li.addEventListener('dragend', () => {
    dragSrcId = null;
    li.classList.remove('dragging');
    clearDragOverStyles();
  });

  li.addEventListener('dragover', (e) => {
    e.preventDefault();
    if (dragSrcId === taskId) return; // 자기 자신 위에서는 무시

    // 마우스 Y 위치로 삽입 방향 결정
    const rect    = li.getBoundingClientRect();
    const midY    = rect.top + rect.height / 2;
    const isBefore = e.clientY < midY;

    clearDragOverStyles();
    li.classList.add(isBefore ? 'drag-over-top' : 'drag-over-bottom');
    li._dropBefore = isBefore;

    e.dataTransfer.dropEffect = 'move';
  });

  li.addEventListener('dragleave', () => {
    li.classList.remove('drag-over-top', 'drag-over-bottom');
  });

  li.addEventListener('drop', (e) => {
    e.preventDefault();
    if (!dragSrcId || dragSrcId === taskId) return;

    reorderTasks(dragSrcId, taskId, li._dropBefore ?? true);
    clearDragOverStyles();
  });
}

/** 모든 항목의 드래그 오버 스타일을 제거한다. */
function clearDragOverStyles() {
  document.querySelectorAll('.task-item').forEach((el) => {
    el.classList.remove('drag-over-top', 'drag-over-bottom');
  });
}

/**
 * srcId 항목을 destId 항목 앞(insertBefore=true) 또는 뒤에 이동한다.
 * 이동 후 정렬 모드를 'manual'로 전환하고 order 필드를 재정규화한다.
 */
function reorderTasks(srcId, destId, insertBefore) {
  const srcIdx  = tasks.findIndex((t) => t.id === srcId);
  const destIdx = tasks.findIndex((t) => t.id === destId);
  if (srcIdx === -1 || destIdx === -1) return;

  // 이동할 항목을 배열에서 꺼냄
  const [moved] = tasks.splice(srcIdx, 1);

  // 목적지 인덱스 재계산 (splice로 인해 인덱스가 바뀔 수 있음)
  const newDestIdx = tasks.findIndex((t) => t.id === destId);
  const insertAt   = insertBefore ? newDestIdx : newDestIdx + 1;
  tasks.splice(insertAt, 0, moved);

  // order 필드 재정규화
  tasks.forEach((t, i) => { t.order = i; });

  // 정렬 모드를 수동으로 전환
  sortBy = 'manual';
  sortSelect.value = 'manual';
  localStorage.setItem(SORT_KEY, 'manual');

  saveTasks();
  renderAll();
}

/* ──────────────────────────────────────────────────────────
   14. 필터
────────────────────────────────────────────────────────── */

function applyFilter(filter, doRender = true) {
  activeFilter = filter;
  localStorage.setItem(FILTER_KEY, filter);

  filterBtns.forEach((btn) => {
    const isActive = btn.dataset.filter === filter;
    btn.classList.toggle('active', isActive);
    btn.setAttribute('aria-pressed', String(isActive));
  });

  if (doRender) renderAll();
}

/* ──────────────────────────────────────────────────────────
   15. visibleTasks: 필터 + 검색 + 정렬
────────────────────────────────────────────────────────── */

/**
 * 현재 필터·검색·정렬 조건을 모두 적용한 할 일 배열을 반환한다.
 * – 정렬이 'manual'이 아닐 경우 완료 항목은 항상 하단에 배치된다.
 */
function visibleTasks() {
  // 1) 카테고리 필터
  let list = activeFilter === 'all'
    ? [...tasks]
    : tasks.filter((t) => t.category === activeFilter);

  // 2) 검색 필터 (대소문자 무관)
  if (searchQuery) {
    const q = searchQuery.toLowerCase();
    list = list.filter((t) => t.text.toLowerCase().includes(q));
  }

  // 3) 정렬
  switch (sortBy) {
    case 'manual':
      list.sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
      break;
    case 'newest':
      list.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
      break;
    case 'oldest':
      list.sort((a, b) => new Date(a.createdAt) - new Date(b.createdAt));
      break;
    case 'category': {
      const ORDER = { work: 0, personal: 1, study: 2 };
      list.sort((a, b) => ORDER[a.category] - ORDER[b.category]);
      break;
    }
    default:
      break;
  }

  // 4) 완료 항목 하단 배치 (수동 정렬 제외)
  if (sortBy !== 'manual') {
    list = [
      ...list.filter((t) => !t.completed),
      ...list.filter((t) =>  t.completed),
    ];
  }

  return list;
}

/* ──────────────────────────────────────────────────────────
   16. 렌더링
────────────────────────────────────────────────────────── */

/**
 * 할 일 목록 전체를 다시 그린다.
 * DocumentFragment를 사용해 DOM 조작을 최소화한다
 * (100개 이상 항목에서도 한 번의 리플로우만 발생).
 */
function renderAll() {
  const list     = visibleTasks();
  const fragment = document.createDocumentFragment();

  list.forEach((task) => fragment.appendChild(createTaskItem(task)));

  // 기존 목록을 지우고 한 번에 삽입
  taskList.innerHTML = '';
  taskList.appendChild(fragment);

  // 빈 상태 메시지
  emptyMessage.classList.toggle('hidden', list.length > 0);

  // 검색 결과 개수 (스크린 리더 전용)
  if (searchQuery) {
    searchStatus.textContent = `검색 결과 ${list.length}개`;
  } else {
    searchStatus.textContent = '';
  }

  updateToolbar();
  updateDashboard();
}

/**
 * 할 일 항목 <li> DOM 요소를 생성해 반환한다.
 * 접근성(aria-label, role)과 드래그 앤 드롭 이벤트도 함께 설정한다.
 */
function createTaskItem(task) {
  const li = document.createElement('li');
  li.className = 'task-item' + (task.completed ? ' completed' : '');
  li.dataset.id = task.id;
  li.setAttribute('aria-label', `${task.completed ? '완료' : '미완료'}: ${task.text}`);

  // ── 카테고리 컬러 바 ──
  const tag = document.createElement('div');
  tag.className = `category-tag ${task.category}`;
  tag.setAttribute('aria-hidden', 'true');

  // ── 체크박스 ──
  const checkbox = document.createElement('input');
  checkbox.type = 'checkbox';
  checkbox.className = 'task-checkbox';
  checkbox.checked = task.completed;
  checkbox.setAttribute(
    'aria-label',
    `${task.text} - ${task.completed ? '완료됨, 클릭하여 미완료로 변경' : '미완료, 클릭하여 완료로 변경'}`
  );
  checkbox.addEventListener('change', () => handleToggle(task.id));

  // ── task-body: 텍스트 + 메타 ──
  const body = document.createElement('div');
  body.className = 'task-body';

  const textSpan = document.createElement('span');
  textSpan.className = 'task-text';
  textSpan.title = '더블클릭하여 수정';
  textSpan.setAttribute('role', 'button');
  textSpan.setAttribute('aria-label', `${task.text}, 더블클릭하여 수정`);
  textSpan.addEventListener('dblclick', () => enterEditMode(task.id));

  // 검색어가 있으면 하이라이트, 없으면 일반 텍스트
  if (searchQuery) {
    textSpan.innerHTML = highlight(task.text, searchQuery);
  } else {
    textSpan.textContent = task.text;
  }

  const meta = document.createElement('div');
  meta.className = 'task-meta';
  meta.setAttribute('aria-hidden', 'true'); // 중복 읽기 방지

  const catLabel = document.createElement('span');
  catLabel.className = `category-label ${task.category}`;
  catLabel.textContent = `${CATEGORIES[task.category].emoji} ${CATEGORIES[task.category].label}`;

  const timeLabel = document.createElement('span');
  timeLabel.textContent = timeAgo(task.createdAt);

  meta.appendChild(catLabel);
  meta.appendChild(timeLabel);
  body.appendChild(textSpan);
  body.appendChild(meta);

  // ── 삭제 버튼 ──
  const delBtn = document.createElement('button');
  delBtn.className = 'delete-btn';
  delBtn.textContent = '✕';
  delBtn.setAttribute('aria-label', `"${truncate(task.text)}" 삭제`);
  delBtn.addEventListener('click', () => handleDelete(task.id));

  li.appendChild(tag);
  li.appendChild(checkbox);
  li.appendChild(body);
  li.appendChild(delBtn);

  // 드래그 앤 드롭 이벤트 부착
  makeDraggable(li, task.id);

  return li;
}

/* ──────────────────────────────────────────────────────────
   17. 툴바 업데이트 (남은 개수 배지 + 완료 삭제 버튼)
────────────────────────────────────────────────────────── */

function updateToolbar() {
  const remaining = tasks.filter((t) => !t.completed).length;
  const completed = tasks.filter((t) =>  t.completed).length;

  // 남은 개수 배지
  if (remaining > 0) {
    remainingBadge.textContent = `${remaining}개 남음`;
    remainingBadge.classList.remove('hidden');
  } else {
    remainingBadge.classList.add('hidden');
  }

  // 완료 삭제 버튼: 완료 항목이 있을 때만 표시
  clearCompletedBtn.classList.toggle('hidden', completed === 0);
}

/* ──────────────────────────────────────────────────────────
   18. 대시보드 업데이트
────────────────────────────────────────────────────────── */

function updateDashboard() {
  const total     = tasks.length;
  const completed = tasks.filter((t) => t.completed).length;
  const pct       = total === 0 ? 0 : Math.round((completed / total) * 100);

  // 전체 진행률
  dashSummary.textContent = `${completed}/${total} 완료 (${pct}%)`;
  progressBar.style.width = `${pct}%`;
  progressTrack.setAttribute('aria-valuenow', String(pct));

  // 응원 메시지: 완료율 구간에 맞는 메시지 선택
  const match = ENCOURAGEMENTS.find((e) => pct >= e.min && pct <= e.max);
  encouragement.textContent = total > 0 && match ? match.msg : '';

  // 오늘 추가된 항목 수
  const todayStr = new Date().toDateString();
  dashTodayCount.textContent = tasks.filter(
    (t) => new Date(t.createdAt).toDateString() === todayStr
  ).length;

  // 카테고리별 미니 통계
  updateCatBar('work',     dashWorkCount,     barWork);
  updateCatBar('personal', dashPersonalCount, barPersonal);
  updateCatBar('study',    dashStudyCount,    barStudy);
}

/** 특정 카테고리의 완료 현황을 카운트 텍스트와 프로그레스 바에 반영한다. */
function updateCatBar(cat, countEl, barEl) {
  const catTasks = tasks.filter((t) => t.category === cat);
  const total    = catTasks.length;
  const done     = catTasks.filter((t) => t.completed).length;
  const pct      = total === 0 ? 0 : Math.round((done / total) * 100);
  countEl.textContent = `${done}/${total}`;
  barEl.style.width   = `${pct}%`;
}

/* ──────────────────────────────────────────────────────────
   19. 데이터 내보내기 / 가져오기
────────────────────────────────────────────────────────── */

/**
 * 현재 할 일 목록을 JSON 파일로 다운로드한다.
 * 파일명: my-tasks-YYYY-MM-DD.json
 */
function exportData() {
  if (tasks.length === 0) {
    showToast('내보낼 할 일이 없습니다.', null, 2500);
    return;
  }

  const json     = JSON.stringify(tasks, null, 2);
  const blob     = new Blob([json], { type: 'application/json' });
  const url      = URL.createObjectURL(blob);
  const dateStr  = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
  const filename = `my-tasks-${dateStr}.json`;

  // 임시 <a> 태그를 통해 다운로드 트리거
  const a = document.createElement('a');
  a.href     = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url); // 메모리 해제

  showToast(`${tasks.length}개 항목을 ${filename}으로 내보냈습니다`, null, 3000);
  announce(`데이터 내보내기 완료: ${tasks.length}개 항목`);
}

/**
 * JSON 파일에서 할 일 목록을 불러온다.
 * 1) 파일 읽기 및 JSON 파싱
 * 2) 스키마 유효성 검사 (text 필드 필수)
 * 3) 사용자 확인 후 현재 데이터를 덮어씀
 * 4) Undo를 위해 기존 데이터를 토스트 액션에 캡처
 */
function importData(file) {
  if (!file) return;

  const reader = new FileReader();

  reader.onload = (e) => {
    try {
      const raw = JSON.parse(e.target.result);

      // 배열 여부 확인
      if (!Array.isArray(raw)) {
        throw new Error('JSON 배열 형식이어야 합니다');
      }

      // 각 항목 정규화 및 유효성 검사
      const normalized = raw
        .map((item, i) => ({
          id:        typeof item.id === 'string' ? item.id : `imported-${Date.now()}-${i}`,
          text:      String(item.text ?? '').trim(),
          category:  ['work', 'personal', 'study'].includes(item.category) ? item.category : 'work',
          completed: Boolean(item.completed),
          createdAt: item.createdAt || new Date().toISOString(),
          order:     typeof item.order === 'number' ? item.order : i,
        }))
        .filter((t) => t.text.length > 0); // 빈 텍스트 제외

      if (normalized.length === 0) {
        throw new Error('유효한 할 일 항목이 없습니다');
      }

      // 사용자 확인
      const ok = window.confirm(
        `📥 가져오기 확인\n\n` +
        `가져올 항목: ${normalized.length}개\n` +
        `현재 항목: ${tasks.length}개 (덮어쓰기)\n\n` +
        `계속 진행할까요? (토스트에서 실행 취소 가능)`
      );
      if (!ok) return;

      // 기존 데이터 백업 (토스트 실행취소용)
      const backup = [...tasks];

      tasks = normalized;
      saveTasks();
      renderAll();

      showToast(
        `${normalized.length}개 항목을 가져왔습니다`,
        {
          label: '실행 취소',
          fn: () => {
            tasks = backup;
            saveTasks();
            renderAll();
            announce('가져오기 취소됨');
          },
        },
        6000
      );
      announce(`${normalized.length}개 항목 가져오기 완료`);

    } catch (err) {
      // 파싱/검증 오류 처리
      showToast(`가져오기 실패: ${err.message}`, null, 4000);
      console.error('[MyTasks] 가져오기 오류:', err);
    }
  };

  reader.onerror = () => {
    showToast('파일을 읽을 수 없습니다.', null, 3000);
  };

  reader.readAsText(file, 'utf-8');
}

/* ──────────────────────────────────────────────────────────
   20. localStorage
────────────────────────────────────────────────────────── */

/**
 * 디바운스된 저장 함수.
 * 빠른 연속 조작(예: 체크박스 빠르게 클릭) 시 저장 빈도를 줄인다.
 * 단, 삭제·추가 등 중요 조작에서는 직접 saveTasks()를 호출한다.
 */
const debouncedSave = debounce(saveTasks, 400);

function saveTasks() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(tasks));
  } catch (e) {
    // localStorage 용량 초과 등 예외 처리
    console.warn('[MyTasks] 저장 실패:', e);
    showToast('저장 공간이 부족합니다. 오래된 항목을 삭제해주세요.', null, 4000);
  }
}

function loadTasks() {
  try {
    const data = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (!Array.isArray(data)) return [];

    // 기존 데이터 마이그레이션:
    // – category 없는 항목 → 'work' 기본값
    // – order 없는 항목 → 배열 인덱스
    return data.map((t, i) => ({
      category: 'work',
      order:    i,
      ...t,
    }));
  } catch (e) {
    console.warn('[MyTasks] 데이터 불러오기 실패:', e);
    return [];
  }
}

/* ──────────────────────────────────────────────────────────
   21. 키보드 단축키
────────────────────────────────────────────────────────── */

function handleShortcut(e) {
  if (!e.altKey) return;

  // 편집 중인 인라인 input에서는 단축키 비활성화
  const active = document.activeElement;
  if (active && (active.classList.contains('edit-input') || active.classList.contains('task-input'))) {
    return;
  }

  switch (e.key.toLowerCase()) {
    case 'n':
      // Alt+N: 새 할 일 입력창 포커스
      e.preventDefault();
      taskInput.focus();
      taskInput.select();
      break;

    case 'd':
      // Alt+D: 다크 모드 토글
      e.preventDefault();
      themeToggle.checked = !themeToggle.checked;
      toggleTheme();
      break;

    case 'z':
      // Alt+Z: 마지막 삭제 실행취소
      e.preventDefault();
      if (undoStack) undoDelete();
      break;

    // Alt+1~4: 카테고리 필터 전환
    case '1': e.preventDefault(); applyFilter('all');      break;
    case '2': e.preventDefault(); applyFilter('work');     break;
    case '3': e.preventDefault(); applyFilter('personal'); break;
    case '4': e.preventDefault(); applyFilter('study');    break;

    default: break;
  }
}

/* ──────────────────────────────────────────────────────────
   22. 이벤트 바인딩
────────────────────────────────────────────────────────── */

// ── 새 할 일 추가 ──
addBtn.addEventListener('click', handleAdd);
taskInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') handleAdd(); });

// ── 카테고리 필터 ──
filterBtns.forEach((btn) => {
  btn.addEventListener('click', () => applyFilter(btn.dataset.filter));
});

// ── 검색: 디바운스로 렌더 빈도 제한 (150ms) ──
const debouncedRender = debounce(renderAll, 150);
searchInput.addEventListener('input', () => {
  searchQuery = searchInput.value.trim();
  searchClear.classList.toggle('hidden', !searchQuery);
  debouncedRender();
});

searchClear.addEventListener('click', () => {
  searchInput.value = '';
  searchQuery = '';
  searchClear.classList.add('hidden');
  searchInput.focus();
  renderAll();
});

// ── 정렬 ──
sortSelect.addEventListener('change', () => {
  sortBy = sortSelect.value;
  localStorage.setItem(SORT_KEY, sortBy);
  renderAll();
  announce(`정렬 기준: ${SORT_MODES[sortBy]}`);
});

// ── 완료 항목 모두 삭제 ──
clearCompletedBtn.addEventListener('click', handleClearCompleted);

// ── 다크모드 토글 ──
themeToggle.addEventListener('change', toggleTheme);

// ── 내보내기 ──
exportBtn.addEventListener('click', exportData);

// ── 가져오기: 버튼 클릭 → 파일 선택 대화상자 ──
importBtn.addEventListener('click', () => importFile.click());
importFile.addEventListener('change', (e) => {
  if (e.target.files.length > 0) {
    importData(e.target.files[0]);
    // 같은 파일을 다시 선택할 수 있도록 값 초기화
    importFile.value = '';
  }
});

// ── 키보드 단축키 ──
document.addEventListener('keydown', handleShortcut);

/* ──────────────────────────────────────────────────────────
   23. 초기화
────────────────────────────────────────────────────────── */

(function init() {
  // 테마 적용
  initTheme();

  // 정렬 select 동기화
  sortSelect.value = sortBy;

  // 필터 적용 (렌더 없이)
  applyFilter(activeFilter, false);

  // 오늘의 격언
  initQuote();

  // 첫 렌더링
  renderAll();

  // 포커스를 입력창으로
  taskInput.focus();
})();
