// Small Windows entry point. Python/CUDA and weights remain in the project.
using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Windows.Forms;

internal static class SmartCameraLauncher
{
    [STAThread]
    private static void Main(string[] args)
    {
        bool owner;
        using (var mutex = new Mutex(true, "Local\\CameraAI.SmartCamera.Launcher", out owner))
        {
            if (!owner)
            {
                MessageBox.Show("Smart Camera est déjà ouverte.", "Smart Camera");
                return;
            }
            try
            {
                string root = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
                string python = null;
                foreach (string environment in new[] { ".venv", ".venv-audit" })
                {
                    string candidate = Path.Combine(root, environment, "Scripts", "python.exe");
                    if (File.Exists(candidate)) { python = candidate; break; }
                }
                if (python == null || !File.Exists(Path.Combine(root, "desktop.py")))
                    throw new InvalidOperationException("Environnement introuvable. Installe requirements-desktop.txt dans .venv puis conserve cet exécutable à côté de desktop.py.");
                var info = new ProcessStartInfo(python);
                info.Arguments = "\"" + Path.Combine(root, "desktop.py") + "\"";
                foreach (string argument in args)
                {
                    if (argument == "--no-camera") info.Arguments += " --no-camera";
                    if (argument == "--low-latency")
                        info.EnvironmentVariables["OPENCV_FFMPEG_CAPTURE_OPTIONS"] =
                            "analyzeduration;100000|probesize;32768|fpsprobesize;0|fflags;nobuffer|flags;low_delay";
                }
                info.WorkingDirectory = root;
                info.UseShellExecute = false;
                info.CreateNoWindow = true;
                info.WindowStyle = ProcessWindowStyle.Hidden;
                info.RedirectStandardError = true;
                info.RedirectStandardOutput = true;
                info.StandardErrorEncoding = Encoding.UTF8;
                info.StandardOutputEncoding = Encoding.UTF8;
                info.EnvironmentVariables["PYTHONUTF8"] = "1";
                info.EnvironmentVariables["GRADIO_ANALYTICS_ENABLED"] = "False";
                string logPath = Path.Combine(root, "launcher.log");
                using (var output = new StreamWriter(logPath, true, Encoding.UTF8))
                using (var child = new Process())
                {
                    object logLock = new object();
                    DataReceivedEventHandler receive = delegate(object sender, DataReceivedEventArgs e) {
                        if (e.Data == null) return;
                        lock (logLock) { output.WriteLine(e.Data); output.Flush(); }
                    };
                    child.StartInfo = info;
                    child.OutputDataReceived += receive;
                    child.ErrorDataReceived += receive;
                    child.Start();
                    child.BeginOutputReadLine();
                    child.BeginErrorReadLine();
                    child.WaitForExit();
                    if (child.ExitCode != 0)
                        MessageBox.Show("L’application s’est arrêtée avec une erreur. Consulte launcher.log dans le dossier de l’application.", "Smart Camera", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
            catch (Exception error)
            {
                MessageBox.Show(error.Message, "Smart Camera", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }
    }
}
