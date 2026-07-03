const script = document.createElement('script');
script.textContent = 'window.__HOLOSPACES_EXTENSION_INSTALLED__ = true;';
(document.head || document.documentElement).appendChild(script);
script.remove();
