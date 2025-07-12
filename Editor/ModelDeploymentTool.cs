using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using SparkTTS.Editor;

namespace LiveTalk.Editor
{
    /// <summary>
    /// Editor tool for deploying LiveTalk models to StreamingAssets.
    /// Analyzes the codebase to determine which models are required and their precision settings,
    /// then copies only the necessary models to optimize build size and deployment.
    /// </summary>
    public class ModelDeploymentTool : EditorWindow
    {
        #region Model Configuration Data

        /// <summary>
        /// Configuration for a specific model including its precision requirements
        /// </summary>
        [Serializable]
        public class ModelConfig
        {
            public string modelName;
            public string relativePath;
            public string precision; // "fp32", "fp16", "int8"
            public string executionProvider; // "CPU", "CoreML", "CUDA"
            public bool isRequired;
            public long fileSize;
            public string fullPath;
            public string source; // "LiveTalk" or "SparkTTS"

            public ModelConfig(string name, string path, string prec, string provider, bool required = true, string source = "LiveTalk")
            {
                modelName = name;
                relativePath = path;
                precision = prec;
                executionProvider = provider;
                isRequired = required;
                this.source = source;
            }
        }

        /// <summary>
        /// Predefined model configurations based on the LiveTalk codebase analysis
        /// </summary>
        private static readonly Dictionary<string, List<ModelConfig>> ModelConfigurations = new()
        {
            ["LivePortrait"] = new List<ModelConfig>
            {
                new("appearance_feature_extractor", "LiveTalk/models/LivePortrait", "fp32", "CoreML"),
                new("motion_extractor", "LiveTalk/models/LivePortrait", "fp32", "CoreML"),
                new("stitching", "LiveTalk/models/LivePortrait", "fp32", "CPU"),
                new("warping_spade", "LiveTalk/models/LivePortrait", "fp16", "CoreML"),
                new("det_10g", "LiveTalk/models/LivePortrait", "fp32", "CoreML"),
                new("2d106det", "LiveTalk/models/LivePortrait", "fp32", "CoreML"),
                new("landmark", "LiveTalk/models/LivePortrait", "fp32", "CoreML"),
                new("stitching_eye", "LiveTalk/models/LivePortrait", "fp32", "CPU"),
                new("stitching_lip", "LiveTalk/models/LivePortrait", "fp32", "CPU"),
            },
            ["MuseTalk"] = new List<ModelConfig>
            {
                new("unet", "LiveTalk/models/MuseTalk", "fp16", "CoreML"),
                new("vae_encoder", "LiveTalk/models/MuseTalk", "fp16", "CoreML"),
                new("vae_decoder", "LiveTalk/models/MuseTalk", "fp16", "CoreML"),
                new("positional_encoding", "LiveTalk/models/MuseTalk", "fp32", "CPU"),
                new("whisper_encoder", "LiveTalk/models/MuseTalk", "fp32", "CPU"),
                new("face_parsing", "LiveTalk/models/MuseTalk", "fp32", "CoreML"),
            }
        };

        #endregion

        #region UI Fields

        [SerializeField] private Vector2 scrollPosition;
        [SerializeField] private bool showAdvancedOptions = false;
        [SerializeField] private bool includeSparkTTS = true;
        [SerializeField] private bool includeLivePortrait = true;
        [SerializeField] private bool includeMuseTalk = true;
        [SerializeField] private bool overwriteExisting = true;
        [SerializeField] private bool createBackup = true;
        [SerializeField] private bool dryRun = false;

        // Model source and destination paths
        private string sourceModelsPath;
        private string streamingAssetsPath;
        private List<ModelConfig> selectedModels;
        private long totalSelectedSize;

        #endregion

        #region Unity Editor Window

        [MenuItem("LiveTalk/Model Deployment Tool")]
        public static void ShowWindow()
        {
            var window = GetWindow<ModelDeploymentTool>("Model Deployment Tool");
            window.minSize = new Vector2(600, 400);
            window.Show();
        }

        private void OnEnable()
        {
            sourceModelsPath = Path.Combine(Application.dataPath, "Models");
            streamingAssetsPath = Application.streamingAssetsPath;
            RefreshModelList();
        }

        private void OnGUI()
        {
            EditorGUILayout.LabelField("LiveTalk Model Deployment Tool", EditorStyles.boldLabel);
            EditorGUILayout.Space();

            DrawPathConfiguration();
            EditorGUILayout.Space();
            
            DrawModelSelection();
            EditorGUILayout.Space();
            
            DrawAdvancedOptions();
            EditorGUILayout.Space();
            
            DrawDeploymentActions();
        }

        #endregion

        #region UI Drawing Methods

        private void DrawPathConfiguration()
        {
            EditorGUILayout.LabelField("Configuration", EditorStyles.boldLabel);
            
            EditorGUI.BeginChangeCheck();
            sourceModelsPath = EditorGUILayout.TextField("Source Models Path:", sourceModelsPath);
            streamingAssetsPath = EditorGUILayout.TextField("Destination Path:", streamingAssetsPath);
            
            if (EditorGUI.EndChangeCheck())
            {
                RefreshModelList();
            }

            // Validation
            if (!Directory.Exists(sourceModelsPath))
            {
                EditorGUILayout.HelpBox($"Source path does not exist: {sourceModelsPath}", MessageType.Error);
            }
            else
            {
                EditorGUILayout.HelpBox($"✓ Source path found with {CountAvailableModels()} models", MessageType.Info);
            }
        }

        private void DrawModelSelection()
        {
            EditorGUILayout.LabelField("Model Selection", EditorStyles.boldLabel);
            
            // Component toggles
            includeSparkTTS = EditorGUILayout.Toggle("Include SparkTTS Models (via SparkTTS package)", includeSparkTTS);
            if (includeSparkTTS)
            {
                EditorGUI.indentLevel++;
                EditorGUILayout.HelpBox("SparkTTS models will be deployed using the SparkTTS-Unity package deployment tool.", MessageType.Info);
                EditorGUI.indentLevel--;
            }
            
            includeLivePortrait = EditorGUILayout.Toggle("Include LivePortrait Models", includeLivePortrait);
            includeMuseTalk = EditorGUILayout.Toggle("Include MuseTalk Models", includeMuseTalk);
            
            EditorGUILayout.Space();
            
            // LiveTalk model details
            if (selectedModels != null && selectedModels.Count > 0)
            {
                EditorGUILayout.LabelField($"LiveTalk Models ({selectedModels.Count}):", EditorStyles.boldLabel);
                EditorGUILayout.LabelField($"Total Size: {FormatFileSize(totalSelectedSize)}");
                
                scrollPosition = EditorGUILayout.BeginScrollView(scrollPosition, GUILayout.Height(200));
                
                foreach (var model in selectedModels.OrderBy(m => m.relativePath).ThenBy(m => m.modelName))
                {
                    DrawModelItem(model);
                }
                
                EditorGUILayout.EndScrollView();
            }
            else if (includeLivePortrait || includeMuseTalk)
            {
                EditorGUILayout.HelpBox("No LiveTalk models selected or available.", MessageType.Warning);
            }
            
            // SparkTTS status
            if (includeSparkTTS)
            {
                EditorGUILayout.Space();
                EditorGUILayout.LabelField("SparkTTS Models:", EditorStyles.boldLabel);
                EditorGUILayout.HelpBox("✓ SparkTTS deployment tool integrated - models will be deployed automatically", MessageType.Info);
            }
        }

        private void DrawModelItem(ModelConfig model)
        {
            EditorGUILayout.BeginHorizontal();
            
            // Model info
            string displayName = $"{model.modelName}";
            if (model.precision != "none")
                displayName += $" ({model.precision.ToUpper()})";
            
            EditorGUILayout.LabelField(displayName, GUILayout.Width(200));
            EditorGUILayout.LabelField(model.relativePath, GUILayout.Width(250));
            EditorGUILayout.LabelField(FormatFileSize(model.fileSize), GUILayout.Width(80));
            
            // Status
            if (File.Exists(model.fullPath))
            {
                EditorGUILayout.LabelField("✓", GUILayout.Width(20));
            }
            else
            {
                EditorGUILayout.LabelField("✗", GUILayout.Width(20));
            }
            
            EditorGUILayout.EndHorizontal();
        }

        private void DrawAdvancedOptions()
        {
            showAdvancedOptions = EditorGUILayout.Foldout(showAdvancedOptions, "Advanced Options");
            
            if (showAdvancedOptions)
            {
                EditorGUI.indentLevel++;
                
                overwriteExisting = EditorGUILayout.Toggle("Overwrite Existing Files", overwriteExisting);
                createBackup = EditorGUILayout.Toggle("Create Backup", createBackup);
                dryRun = EditorGUILayout.Toggle("Dry Run (Preview Only)", dryRun);
                
                if (dryRun)
                {
                    EditorGUILayout.HelpBox("Dry run mode: No files will be copied, only operations will be logged.", MessageType.Info);
                }
                
                EditorGUI.indentLevel--;
            }
        }

        private void DrawDeploymentActions()
        {
            EditorGUILayout.LabelField("Deployment", EditorStyles.boldLabel);
            
            EditorGUILayout.BeginHorizontal();
            
            if (GUILayout.Button("Refresh Model List", GUILayout.Height(30)))
            {
                RefreshModelList();
            }
            
            bool hasSelection = includeSparkTTS || includeLivePortrait || includeMuseTalk;
            EditorGUI.BeginDisabledGroup(!hasSelection);
            
            string buttonText = "Deploy Selected Models";
            if (includeSparkTTS && (includeLivePortrait || includeMuseTalk))
                buttonText = "Deploy All Models";
            else if (includeSparkTTS)
                buttonText = "Deploy SparkTTS Models";
            else if (includeLivePortrait || includeMuseTalk)
                buttonText = "Deploy LiveTalk Models";
            
            if (GUILayout.Button(buttonText, GUILayout.Height(30)))
            {
                DeployModels();
            }
            
            EditorGUI.EndDisabledGroup();
            
            EditorGUILayout.EndHorizontal();
            
            if (!hasSelection)
            {
                EditorGUILayout.HelpBox("Please select at least one model category to deploy.", MessageType.Info);
            }
            
            if (GUILayout.Button("Clean StreamingAssets", GUILayout.Height(25)))
            {
                CleanStreamingAssets();
            }
        }

        #endregion

        #region Model Management

        private void RefreshModelList()
        {
            selectedModels = new List<ModelConfig>();
            totalSelectedSize = 0;
            
            // Add LiveTalk models
            if (includeLivePortrait)
                AddModelsFromCategory("LivePortrait");
            
            if (includeMuseTalk)
                AddModelsFromCategory("MuseTalk");
            
            // Add SparkTTS models if selected
            if (includeSparkTTS)
                AddSparkTTSModels();
            
            // Calculate file sizes and validate paths
            foreach (var model in selectedModels)
            {
                if (model.source == "LiveTalk")
                {
                    UpdateModelInfo(model);
                }
                else if (model.source == "SparkTTS")
                {
                    UpdateSparkTTSModelInfo(model);
                }
            }
            
            Repaint();
        }

        private void AddModelsFromCategory(string category)
        {
            if (ModelConfigurations.ContainsKey(category))
            {
                selectedModels.AddRange(ModelConfigurations[category]);
            }
        }

        private void AddSparkTTSModels()
        {
            try
            {
                // Get SparkTTS model configurations from the SparkTTS deployment tool
                var sparkTTSConfigs = GetSparkTTSModelConfigurations();
                
                foreach (var config in sparkTTSConfigs)
                {
                    var liveTalkConfig = new ModelConfig(
                        config.modelName,
                        config.relativePath,
                        config.precision,
                        "CPU", // Default execution provider for SparkTTS models
                        config.isRequired,
                        "SparkTTS"
                    );
                    selectedModels.Add(liveTalkConfig);
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[LiveTalk] Could not load SparkTTS models: {ex.Message}");
            }
        }

        private List<SparkTTS.Editor.ModelDeploymentTool.ModelConfig> GetSparkTTSModelConfigurations()
        {
            var configs = new List<SparkTTS.Editor.ModelDeploymentTool.ModelConfig>();
            
            // Get SparkTTS model configurations using the public API
            configs.AddRange(SparkTTS.Editor.ModelDeploymentTool.GetModelConfigurations("SparkTTS"));
            configs.AddRange(SparkTTS.Editor.ModelDeploymentTool.GetModelConfigurations("SparkTTS_LLM"));
            
            return configs;
        }

        private void UpdateSparkTTSModelInfo(ModelConfig model)
        {
            // Create a SparkTTS ModelConfig to use with the SparkTTS tool
            var sparkTTSModel = new SparkTTS.Editor.ModelDeploymentTool.ModelConfig(
                model.modelName,
                model.relativePath,
                model.precision,
                model.isRequired
            );
            
            // Use SparkTTS tool to update model info
            sparkTTSModel = SparkTTS.Editor.ModelDeploymentTool.UpdateModelInfo(sparkTTSModel, sourceModelsPath);
            
            // Copy updated values back to our model
            model.fullPath = sparkTTSModel.fullPath;
            model.fileSize = sparkTTSModel.fileSize;
            totalSelectedSize += model.fileSize;
        }

        private void UpdateModelInfo(ModelConfig model)
        {
            // Build the full path based on precision
            string fileName = model.modelName;
            string extension = ".onnx";
            
            // Handle special cases
            if (model.modelName.Contains(".json") || model.modelName.Contains(".txt"))
            {
                fileName = model.modelName;
                extension = "";
            }
            else if (model.precision == "fp16")
            {
                fileName += "_fp16";
            }
            else if (model.precision == "int8")
            {
                fileName += "_int8";
            }
            
            // Handle special case for UNet model with data file
            if (model.modelName == "unet" && model.precision == "fp32")
            {
                // Check for .onnx.data file
                string dataFileName = fileName + ".onnx.data";
                string dataPath = Path.Combine(sourceModelsPath, model.relativePath, dataFileName);
                if (File.Exists(dataPath))
                {
                    // This is the large unet model with separate data file
                    model.fullPath = Path.Combine(sourceModelsPath, model.relativePath, fileName + extension);
                    if (File.Exists(model.fullPath))
                    {
                        model.fileSize = new FileInfo(model.fullPath).Length + new FileInfo(dataPath).Length;
                        totalSelectedSize += model.fileSize;
                    }
                    return;
                }
            }
            
            model.fullPath = Path.Combine(sourceModelsPath, model.relativePath, fileName + extension);
            
            if (File.Exists(model.fullPath))
            {
                model.fileSize = new FileInfo(model.fullPath).Length;
                totalSelectedSize += model.fileSize;
            }
            else
            {
                model.fileSize = 0;
                Debug.LogWarning($"Model file not found: {model.fullPath}");
            }
        }

        private int CountAvailableModels()
        {
            if (!Directory.Exists(sourceModelsPath))
                return 0;
            
            int count = 0;
            string[] searchPatterns = { "*.onnx", "*.json", "*.txt" };
            
            foreach (var pattern in searchPatterns)
            {
                count += Directory.GetFiles(sourceModelsPath, pattern, SearchOption.AllDirectories).Length;
            }
            
            return count;
        }

        #endregion

        #region Deployment Operations

        private void DeployModels()
        {
            // Validate that at least one component is selected
            if (!includeSparkTTS && !includeLivePortrait && !includeMuseTalk)
            {
                EditorUtility.DisplayDialog("No Models Selected", "Please select at least one model category to deploy.", "OK");
                return;
            }
            
            bool liveTalkSuccess = true;
            bool sparkTTSSuccess = true;
            int totalDeployed = 0;

            // Deploy LiveTalk models
            if ((includeLivePortrait || includeMuseTalk) && selectedModels != null && selectedModels.Count > 0)
            {
                var missingModels = selectedModels.Where(m => !File.Exists(m.fullPath)).ToList();
                if (missingModels.Any())
                {
                    string missingList = string.Join("\n", missingModels.Select(m => $"- {m.modelName} at {m.fullPath}"));
                    EditorUtility.DisplayDialog("Missing Models", $"The following LiveTalk models are missing:\n{missingList}", "OK");
                    return;
                }

                if (!dryRun)
                {
                    bool proceed = EditorUtility.DisplayDialog("Deploy Models", 
                        $"Deploy {selectedModels.Count} LiveTalk models ({FormatFileSize(totalSelectedSize)}) to StreamingAssets?", 
                        "Deploy", "Cancel");
                    
                    if (!proceed)
                        return;
                }

                liveTalkSuccess = DeployLiveTalkModels();
                if (liveTalkSuccess)
                {
                    totalDeployed += selectedModels.Count;
                }
            }

            // Deploy SparkTTS models if selected
            if (includeSparkTTS)
            {
                sparkTTSSuccess = DeploySparkTTSModels();
            }

            // Show final result
            if (!dryRun && (liveTalkSuccess || sparkTTSSuccess))
            {
                AssetDatabase.Refresh();
                
                string message = "Deployment completed:\n";
                if (liveTalkSuccess && selectedModels?.Count > 0)
                    message += $"• LiveTalk: {selectedModels.Count} models deployed\n";
                if (sparkTTSSuccess)
                    message += $"• SparkTTS: Models deployed via SparkTTS package\n";
                
                EditorUtility.DisplayDialog("Deployment Complete", message, "OK");
            }
            else if (!liveTalkSuccess || !sparkTTSSuccess)
            {
                EditorUtility.DisplayDialog("Deployment Error", "Some deployments failed. Check console for details.", "OK");
            }
        }

        private bool DeployLiveTalkModels()
        {
            try
            {
                int progress = 0;
                int total = selectedModels.Count;
                
                foreach (var model in selectedModels)
                {
                    if (!dryRun)
                    {
                        EditorUtility.DisplayProgressBar("Deploying LiveTalk Models", $"Copying {model.modelName}...", (float)progress / total);
                    }
                    
                    DeployModel(model);
                    progress++;
                }
                
                if (!dryRun)
                {
                    EditorUtility.ClearProgressBar();
                }
                else
                {
                    Debug.Log($"[LiveTalk DRY RUN] Would have deployed {selectedModels.Count} models");
                }
                
                return true;
            }
            catch (Exception ex)
            {
                if (!dryRun)
                {
                    EditorUtility.ClearProgressBar();
                }
                Debug.LogError($"[LiveTalk] Model deployment failed: {ex}");
                return false;
            }
        }

        private bool DeploySparkTTSModels()
        {
            try
            {
                // Create SparkTTS deployment options
                var options = new SparkTTS.Editor.ModelDeploymentTool.DeploymentOptions
                {
                    OverwriteExisting = overwriteExisting,
                    CreateBackup = createBackup,
                    DryRun = dryRun,
                    IncludeSparkTTS = true,
                    IncludeLLM = true
                };

                // Call SparkTTS deployment
                string sparkTTSSourcePath = Path.Combine(Application.dataPath, "Models");
                string sparkTTSDestPath = Application.streamingAssetsPath;
                
                bool success = SparkTTS.Editor.ModelDeploymentTool.DeploySparkTTSModels(
                    sparkTTSSourcePath, 
                    sparkTTSDestPath, 
                    options);
                
                if (success)
                {
                    Debug.Log("[LiveTalk] SparkTTS models deployed successfully via SparkTTS package.");
                }
                else
                {
                    Debug.LogError("[LiveTalk] SparkTTS deployment failed.");
                }
                
                return success;
            }
            catch (Exception ex)
            {
                Debug.LogError($"[LiveTalk] Failed to deploy SparkTTS models: {ex.Message}");
                return false;
            }
        }

        private void DeployModel(ModelConfig model)
        {
            string destinationDir = Path.Combine(streamingAssetsPath, model.relativePath);
            string destinationPath = Path.Combine(destinationDir, Path.GetFileName(model.fullPath));
            
            if (dryRun)
            {
                Debug.Log($"[DRY RUN] Would copy: {model.fullPath} -> {destinationPath}");
                
                // Check for additional files (like .onnx.data)
                if (model.modelName == "unet" && model.precision == "fp32")
                {
                    string dataSourcePath = model.fullPath + ".data";
                    string dataDestPath = destinationPath + ".data";
                    if (File.Exists(dataSourcePath))
                    {
                        Debug.Log($"[DRY RUN] Would copy: {dataSourcePath} -> {dataDestPath}");
                    }
                }
                return;
            }

            // Create destination directory
            Directory.CreateDirectory(destinationDir);
            
            // Create backup if requested
            if (createBackup && File.Exists(destinationPath))
            {
                string backupPath = destinationPath + ".backup";
                File.Copy(destinationPath, backupPath, true);
                Debug.Log($"Created backup: {backupPath}");
            }
            
            // Copy the main model file
            File.Copy(model.fullPath, destinationPath, overwriteExisting);
            Debug.Log($"Copied: {model.fullPath} -> {destinationPath}");
            
            // Handle special case for unet model with data file
            if (model.modelName == "unet" && model.precision == "fp32")
            {
                string dataSourcePath = model.fullPath + ".data";
                string dataDestPath = destinationPath + ".data";
                if (File.Exists(dataSourcePath))
                {
                    // Create backup for data file if requested
                    if (createBackup && File.Exists(dataDestPath))
                    {
                        string backupDataPath = dataDestPath + ".backup";
                        File.Copy(dataDestPath, backupDataPath, true);
                        Debug.Log($"Created backup: {backupDataPath}");
                    }
                    
                    File.Copy(dataSourcePath, dataDestPath, overwriteExisting);
                    Debug.Log($"Copied: {dataSourcePath} -> {dataDestPath}");
                }
            }
        }

        private void CleanStreamingAssets()
        {
            if (!Directory.Exists(streamingAssetsPath))
            {
                EditorUtility.DisplayDialog("Nothing to Clean", "StreamingAssets LiveTalk directory does not exist.", "OK");
                return;
            }

            bool proceed = EditorUtility.DisplayDialog("Clean StreamingAssets", 
                "This will delete all LiveTalk models from StreamingAssets. Continue?", 
                "Delete", "Cancel");
            
            if (!proceed)
                return;

            try
            {
                Directory.Delete(streamingAssetsPath, true);
                AssetDatabase.Refresh();
                
                EditorUtility.DisplayDialog("Clean Complete", "StreamingAssets LiveTalk directory has been cleaned.", "OK");
                Debug.Log("StreamingAssets LiveTalk directory cleaned");
            }
            catch (Exception ex)
            {
                EditorUtility.DisplayDialog("Clean Error", $"Error cleaning StreamingAssets: {ex.Message}", "OK");
                Debug.LogError($"Clean operation failed: {ex}");
            }
        }

        #endregion

        #region Utility Methods

        private string FormatFileSize(long bytes)
        {
            if (bytes == 0) return "0 B";
            
            string[] sizes = { "B", "KB", "MB", "GB", "TB" };
            int order = 0;
            double size = bytes;
            
            while (size >= 1024 && order < sizes.Length - 1)
            {
                order++;
                size /= 1024;
            }
            
            return $"{size:0.##} {sizes[order]}";
        }

        #endregion
    }
} 