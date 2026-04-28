const STORY_ID = document.querySelector('#story-grid').dataset.storyId;

async function postJSON(url, body) {
  const resp = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: body ? JSON.stringify(body) : null,
  });
  if (!resp.ok) {
    const detail = await resp.text();
    alert(`Error ${resp.status}: ${detail}`);
    return null;
  }
  return await resp.json();
}

function updateRollup(rollup) {
  document.querySelector('#rollup-pending').textContent  = `Pending: ${rollup.pending}`;
  document.querySelector('#rollup-approved').textContent = `Approved: ${rollup.approved}`;
  document.querySelector('#rollup-rejected').textContent = `Rejected: ${rollup.rejected}`;
  document.querySelector('#rollup-regen').textContent    = `Regen: ${rollup.regen_requested}`;
}

function updateSceneTile(sceneId, sceneJson) {
  const tile = document.querySelector(`.scene-tile[data-scene-id="${sceneId}"]`);
  if (!tile) return;
  tile.dataset.status = sceneJson.review_status;
  const pill = tile.querySelector('.status-pill');
  pill.className = `status-pill ${sceneJson.review_status}`;
  pill.textContent = sceneJson.review_status;
  const header = tile.querySelector('.scene-header');
  let badge = header.querySelector('.regen-badge');
  if (sceneJson.regen_count > 0) {
    if (!badge) {
      badge = document.createElement('span');
      badge.className = 'regen-badge';
      badge.title = 'regen attempts';
      header.appendChild(badge);
    }
    badge.textContent = `↻${sceneJson.regen_count}`;
  } else if (badge) {
    badge.remove();
  }
}

document.addEventListener('click', async (ev) => {
  const btn = ev.target.closest('button');
  if (!btn) return;
  const sceneId = btn.dataset.scene;

  if (btn.classList.contains('approve')) {
    const data = await postJSON(`/api/story/${STORY_ID}/scene/${sceneId}/approve`);
    if (data) { updateRollup(data.rollup); updateSceneTile(sceneId, data.scene); }
  }
  else if (btn.classList.contains('regenerate')) {
    const data = await postJSON(`/api/story/${STORY_ID}/scene/${sceneId}/regenerate`);
    if (data) { updateRollup(data.rollup); updateSceneTile(sceneId, data.scene); }
  }
  else if (btn.classList.contains('reject')) {
    const dlg = document.querySelector('#reject-dialog');
    document.querySelector('#reject-scene-id').textContent = sceneId;
    document.querySelector('#reject-feedback').value = '';
    dlg.dataset.sceneId = sceneId;
    dlg.showModal();
  }
  else if (btn.classList.contains('alt-thumb')) {
    const candidateIdx = parseInt(btn.dataset.candidate, 10);
    const data = await postJSON(
      `/api/story/${STORY_ID}/scene/${btn.dataset.scene}/select_candidate`,
      {index: candidateIdx},
    );
    if (data) {
      window.location.reload();
    }
  }
  else if (btn.id === 'batch-approve-btn') {
    const threshold = parseFloat(document.querySelector('#batch-threshold').value);
    if (!isFinite(threshold)) { alert('threshold must be a number'); return; }
    const data = await postJSON(`/api/story/${STORY_ID}/batch_approve`, {threshold});
    if (data) {
      window.location.reload();
    }
  }
  else if (btn.id === 'reject-submit-btn') {
    ev.preventDefault();
    const dlg = document.querySelector('#reject-dialog');
    const sceneId = dlg.dataset.sceneId;
    const feedback = document.querySelector('#reject-feedback').value.trim();
    if (!feedback) { alert('feedback required'); return; }
    const data = await postJSON(
      `/api/story/${STORY_ID}/scene/${sceneId}/reject`,
      {feedback},
    );
    dlg.close();
    if (data) { updateRollup(data.rollup); updateSceneTile(sceneId, data.scene); }
  }
});
