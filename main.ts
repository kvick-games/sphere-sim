import { startWebGPUApp } from './src/webgpu/app';

startWebGPUApp().catch((error) => {
  console.error(error);
  const app = document.getElementById('app') ?? document.body;
  const panel = document.createElement('div');
  panel.className = 'fatal-webgpu';
  panel.textContent = error instanceof Error ? error.message : String(error);
  app.appendChild(panel);
});
