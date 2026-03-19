const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let img = new Image();
img.crossOrigin = 'anonymous';

let boxes = [];
let currentClass = 'object';
let classes = new Map();  // id -> name
let isDrawing = false;
let startX, startY;

img.onload = function() {
    const maxW = 900, maxH = 600;
    let w = img.width, h = img.height;
    if (w > maxW || h > maxH) {
        const r = Math.min(maxW/w, maxH/h);
        w = Math.floor(img.width * r);
        h = Math.floor(img.height * r);
    }
    canvas.width = w;
    canvas.height = h;
    draw();
    loadAnnotations();
};

img.src = imageUrl;

function loadClasses() {
    fetch(`/projects/${projectId}/annotate/${imageId}/classes`)
        .then(r => r.json())
        .then(data => {
            data.forEach(c => classes.set(c.id, c.name));
            if (classes.size === 0) classes.set(0, 'object');
            renderClassList();
        });
}

function loadAnnotations() {
    fetch(`/projects/${projectId}/annotate/${imageId}/annotations`)
        .then(r => r.json())
        .then(data => {
            boxes = data.map(b => ({
                ...b,
                x1: (b.x_center - b.width/2) * canvas.width,
                y1: (b.y_center - b.height/2) * canvas.height,
                x2: (b.x_center + b.width/2) * canvas.width,
                y2: (b.y_center + b.height/2) * canvas.height
            }));
            loadClasses();
            draw();
        });
}

function renderClassList() {
    const el = document.getElementById('classList');
    el.innerHTML = Array.from(classes.entries()).map(([id, name]) => 
        `<span class="badge bg-secondary me-1 mb-1" style="cursor:pointer" onclick="currentClass='${name}'">${name}</span>`
    ).join('');
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    
    boxes.forEach((b, i) => {
        const x1 = (b.x_center - b.width/2) * canvas.width;
        const y1 = (b.y_center - b.height/2) * canvas.height;
        const w = b.width * canvas.width;
        const h = b.height * canvas.height;
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, w, h);
        ctx.fillStyle = 'rgba(0,255,0,0.2)';
        ctx.fillRect(x1, y1, w, h);
        ctx.fillStyle = '#00ff00';
        ctx.font = '12px sans-serif';
        ctx.fillText(b.class_name, x1, y1 - 2);
    });
    
    if (isDrawing) {
        ctx.strokeStyle = '#ff0000';
        ctx.lineWidth = 2;
        ctx.strokeRect(startX, startY, (canvas.width/canvas.width) * (event?.offsetX - startX || 0), (canvas.height/canvas.height) * (event?.offsetY - startY || 0));
    }
    
    document.getElementById('annotationList').innerHTML = boxes.map((b, i) => 
        `<div class="d-flex justify-content-between py-1"><span>${b.class_name}</span><button class="btn btn-sm btn-link text-danger p-0" onclick="removeBox(${i})">×</button></div>`
    ).join('');
}

canvas.addEventListener('mousedown', e => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    startX = (e.clientX - rect.left) * scaleX;
    startY = (e.clientY - rect.top) * scaleY;
    isDrawing = true;
});

canvas.addEventListener('mousemove', e => {
    if (!isDrawing) return;
    draw();
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    ctx.strokeStyle = '#ff0000';
    ctx.lineWidth = 2;
    ctx.strokeRect(Math.min(startX, x), Math.min(startY, y), Math.abs(x - startX), Math.abs(y - startY));
});

canvas.addEventListener('mouseup', e => {
    if (!isDrawing) return;
    isDrawing = false;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const endX = (e.clientX - rect.left) * scaleX;
    const endY = (e.clientY - rect.top) * scaleY;
    const x1 = Math.min(startX, endX), x2 = Math.max(startX, endX);
    const y1 = Math.min(startY, endY), y2 = Math.max(startY, endY);
    if (x2 - x1 > 5 && y2 - y1 > 5) {
        const x_center = (x1 + x2) / 2 / canvas.width;
        const y_center = (y1 + y2) / 2 / canvas.height;
        const width = (x2 - x1) / canvas.width;
        const height = (y2 - y1) / canvas.height;
        const cls = document.getElementById('classInput').value.trim() || currentClass;
        let cid = [...classes.entries()].find(([_,n]) => n === cls)?.[0];
        if (cid === undefined) {
            cid = classes.size ? Math.max(...classes.keys()) + 1 : 0;
            classes.set(cid, cls);
        }
        boxes.push({ class_id: cid, class_name: cls, x_center, y_center, width, height });
        renderClassList();
    }
    draw();
});

document.getElementById('classInput').addEventListener('keydown', e => {
    if (e.key === 'Enter') {
        const v = e.target.value.trim();
        if (v && ![...classes.values()].includes(v)) {
            const cid = classes.size ? Math.max(...classes.keys()) + 1 : 0;
            classes.set(cid, v);
            currentClass = v;
            renderClassList();
        }
    }
});

function removeBox(i) {
    boxes.splice(i, 1);
    draw();
}

document.getElementById('saveBtn').addEventListener('click', () => {
    const data = boxes.map(b => ({
        class_id: b.class_id,
        class_name: b.class_name,
        x_center: b.x_center,
        y_center: b.y_center,
        width: b.width,
        height: b.height
    }));
    fetch(`/projects/${projectId}/annotate/${imageId}/annotations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })
    .then(r => r.json())
    .then(d => { if (d.success) alert('Anotasi tersimpan!'); })
    .catch(e => alert('Gagal menyimpan: ' + e));
});
