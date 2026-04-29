// S7.1.B6.4 -- character gallery Pick button handler.
// Vanilla fetch; on click, POSTs the character + chosen path to the backend
// and mutates the DOM so the picked thumbnail's button shows "Picked" + is
// disabled, and any previously-picked thumbnail in the same row goes back
// to "Pick this one".

const STORY_ID = document.querySelector('#character-grid').dataset.storyId;

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

function updateRowAfterPick(characterName, pickedPath) {
  const row = document.querySelector(
    `.character-row[data-character="${characterName}"]`,
  );
  if (!row) return;

  // Update the header pill.
  const header = row.querySelector('.character-header');
  let pill = header.querySelector('.picked-pill, .unpicked-pill');
  if (pill) pill.remove();
  const newPill = document.createElement('span');
  newPill.className = 'picked-pill';
  newPill.dataset.pickedPath = pickedPath;
  newPill.textContent = `Picked: ${pickedPath}`;
  header.appendChild(newPill);

  // Reset all buttons in this row, then mark the picked one.
  row.querySelectorAll('.pick-button').forEach((btn) => {
    const label = btn.querySelector('.pick-label');
    if (btn.dataset.path === pickedPath) {
      btn.disabled = true;
      if (label) label.textContent = 'Picked';
    } else {
      btn.disabled = false;
      if (label) label.textContent = 'Pick this one';
    }
  });
}

document.addEventListener('click', async (ev) => {
  const btn = ev.target.closest('.pick-button');
  if (!btn) return;
  if (btn.disabled) return;

  const character = btn.dataset.character;
  const path = btn.dataset.path;
  if (!character || !path) return;

  const data = await postJSON(
    `/api/story/${STORY_ID}/select_character_reference`,
    {character, path},
  );
  if (data) {
    updateRowAfterPick(character, path);
  }
});
