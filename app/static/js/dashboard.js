document.addEventListener('DOMContentLoaded', function () {
    // === FILE INPUT HANDLER ===
    const fileInput = document.getElementById('file-input');
    const fileSelected = document.getElementById('file-selected');
    if (fileInput && fileSelected) {
        fileInput.addEventListener('change', function () {
            const fileName = this.files[0] ? this.files[0].name : '';
            fileSelected.textContent = fileName;
        });
    }

    // === MODAL HANDLER (SUMMARY) ===
    const summaryModal = document.getElementById('summaryModal');
    const summaryBtn = document.getElementById('summaryBtn');
    const summaryClose = summaryModal?.querySelector('.close');

    if (summaryBtn && summaryModal) {
        summaryBtn.addEventListener('click', function () {
            summaryModal.style.display = 'block';
        });
    }

    if (summaryClose) {
        summaryClose.addEventListener('click', function () {
            summaryModal.style.display = 'none';
        });
    }

    window.addEventListener('click', function (event) {
        if (event.target === summaryModal) {
            summaryModal.style.display = 'none';
        }
    });
});