const { app, BrowserWindow } = require('electron');
const path = require('path');
const url = require('url');

function createWindow() {
  // Create the browser window.
  const mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    autoHideMenuBar: true,
    title: "SkyWatch Tracker",
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      webSecurity: false // IMPORTANT: Allows connecting to insecure HTTP MJPEG streams without CORS/Mixed Content issues
    }
  });

  // Determines whether we are in dev or prod mode based on args or missing packager
  const isDev = process.argv.includes('--dev');

  if (isDev) {
    // In dev, load Vite's localhost
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    // In production, load the built HTML bundle
    mainWindow.loadURL(url.format({
      pathname: path.join(__dirname, 'dist', 'index.html'),
      protocol: 'file:',
      slashes: true
    }));
  }
}

app.commandLine.appendSwitch('ignore-certificate-errors'); // Bypass cert issues for local cameras

app.whenReady().then(() => {
  createWindow();

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit();
});
