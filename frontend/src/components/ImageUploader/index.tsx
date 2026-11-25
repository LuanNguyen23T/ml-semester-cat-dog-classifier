import { useState, useRef, useCallback, useEffect } from 'react';
import { Upload, Clipboard } from 'lucide-react';

interface ImageUploaderProps {
  onFileSelected: (file: File) => void;
}

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const ACCEPTED_EXTENSIONS = /\.(jpe?g|png|webp)$/i;
const MAX_SIZE_MB = 10;

export function ImageUploader({ onFileSelected }: ImageUploaderProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const validateAndSet = useCallback(
    (file: File) => {
      setValidationError(null);

      const hasValidMime = ACCEPTED_TYPES.includes(file.type);
      const hasValidExt = ACCEPTED_EXTENSIONS.test(file.name);
      if (!hasValidMime && !(file.type === '' && hasValidExt)) {
        setValidationError('Sai loại file: chỉ hỗ trợ ảnh JPG, PNG hoặc WebP.');
        return;
      }

      if (file.size > MAX_SIZE_MB * 1024 * 1024) {
        setValidationError(`Ảnh vượt quá dung lượng tối đa ${MAX_SIZE_MB} MB. Vui lòng chọn ảnh nhỏ hơn.`);
        return;
      }

      onFileSelected(file);
    },
    [onFileSelected],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) validateAndSet(file);
    },
    [validateAndSet],
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) validateAndSet(file);
      e.target.value = '';
    },
    [validateAndSet],
  );

  const handleClipboardEvent = useCallback(
    (clipboardData: DataTransfer | null) => {
      if (!clipboardData) return false;

      // 1. Check for files in clipboardData.files
      if (clipboardData.files && clipboardData.files.length > 0) {
        const file = clipboardData.files[0];
        if (file.type.startsWith('image/')) {
          validateAndSet(file);
          return true;
        }
      }

      // 2. Check items in clipboardData.items (for screenshots, snipping tool, copy image)
      if (clipboardData.items) {
        const items = Array.from(clipboardData.items);
        const imageItem = items.find((item) => item.type.startsWith('image/'));
        if (imageItem) {
          const blob = imageItem.getAsFile();
          if (blob) {
            const ext = blob.type.includes('png') ? 'png' : blob.type.includes('webp') ? 'webp' : 'jpg';
            const fileName =
              blob.name && blob.name !== 'image.png'
                ? blob.name
                : `pasted_image_${Date.now()}.${ext}`;
            const file = new File([blob], fileName, { type: blob.type || 'image/png' });
            validateAndSet(file);
            return true;
          }
        }
      }

      return false;
    },
    [validateAndSet],
  );

  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      const activeEl = document.activeElement;
      // Don't intercept paste if user is typing in a text input or textarea
      if (
        activeEl &&
        (activeEl.tagName === 'INPUT' || activeEl.tagName === 'TEXTAREA' || activeEl.getAttribute('contenteditable') === 'true')
      ) {
        return;
      }

      const handled = handleClipboardEvent(e.clipboardData);
      if (handled) {
        e.preventDefault();
      }
    };

    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, [handleClipboardEvent]);

  return (
    <div style={{ width: '100%' }}>
      <div
        className={`upload-zone ${isDragOver ? 'drag-over' : ''}`}
        tabIndex={0}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onPaste={(e) => {
          const handled = handleClipboardEvent(e.clipboardData);
          if (handled) e.preventDefault();
        }}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        style={{ cursor: 'pointer', padding: '56px 20px', outline: 'none' }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(',')}
          onChange={handleChange}
          style={{ display: 'none' }}
        />

        <div style={{ position: 'relative', zIndex: 1 }}>
          <div
            className="animate-float"
            style={{
              width: 72,
              height: 72,
              borderRadius: '50%',
              background: 'rgba(139,92,246,0.15)',
              border: '1px solid rgba(139,92,246,0.3)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 24px',
            }}
          >
            <Upload size={32} color="var(--color-accent)" />
          </div>

          <h3
            style={{
              fontSize: 20,
              fontWeight: 600,
              color: 'var(--color-text)',
              marginBottom: 8,
            }}
          >
            {isDragOver ? 'Thả ảnh vào đây!' : 'Kéo & thả hoặc Dán (Ctrl+V / ⌘V) ảnh vào đây'}
          </h3>

          <p style={{ color: 'var(--color-text-muted)', fontSize: 14, marginBottom: 20 }}>
            hoặc{' '}
            <span style={{ color: 'var(--color-accent)', fontWeight: 600 }}>
              chọn từ thư viện / chụp ảnh màn hình rồi dán
            </span>
          </p>

          <div
            style={{
              display: 'flex',
              gap: 8,
              justifyContent: 'center',
              flexWrap: 'wrap',
            }}
          >
            {['JPG', 'PNG', 'WebP'].map((fmt) => (
              <span
                key={fmt}
                style={{
                  padding: '4px 10px',
                  background: 'rgba(139,92,246,0.1)',
                  border: '1px solid rgba(139,92,246,0.2)',
                  borderRadius: 6,
                  fontSize: 12,
                  color: 'var(--color-text-muted)',
                  fontWeight: 500,
                }}
              >
                {fmt}
              </span>
            ))}
            <span
              style={{
                padding: '4px 10px',
                background: 'rgba(16, 185, 129, 0.1)',
                border: '1px solid rgba(16, 185, 129, 0.25)',
                borderRadius: 6,
                fontSize: 12,
                color: '#059669',
                fontWeight: 600,
                display: 'inline-flex',
                alignItems: 'center',
                gap: 5,
              }}
            >
              <Clipboard size={12} /> Hỗ trợ Paste Ctrl+V / ⌘V
            </span>
            <span
              style={{
                padding: '4px 10px',
                background: 'rgba(139,92,246,0.1)',
                border: '1px solid rgba(139,92,246,0.2)',
                borderRadius: 6,
                fontSize: 12,
                color: 'var(--color-text-muted)',
                fontWeight: 500,
              }}
            >
              Tối đa {MAX_SIZE_MB} MB
            </span>
          </div>

          {validationError && (
            <p
              role="alert"
              style={{ color: 'var(--color-error)', fontSize: 13, marginTop: 16, fontWeight: 600 }}
            >
              ⚠ {validationError}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
