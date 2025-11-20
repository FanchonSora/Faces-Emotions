using UnityEngine;
using System;
using System.Threading.Tasks;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using Newtonsoft.Json;

namespace StarterAssets
{
    public class EmotionClient : MonoBehaviour
    {
        [Header("Server Settings")]
        public string serverUrl = "ws://localhost:8000/ws";
        
        [Header("Camera Settings")]
        public int cameraWidth = 640;
        public int cameraHeight = 480;
        public int targetFPS = 10;
        
        private ClientWebSocket webSocket; 
        private WebCamTexture webcamTexture;
        private StarterAssetsInputs input;
        private float lastSendTime;
        private float sendInterval;
        private bool isConnected = false;
        
        private void Start()
        {
            input = GetComponent<StarterAssetsInputs>();
            sendInterval = 1f / targetFPS;
            
            webcamTexture = new WebCamTexture(cameraWidth, cameraHeight);
            webcamTexture.Play();
            
            _ = ConnectWebSocket();
        }
        
        private async Task ConnectWebSocket()
        {
            try
            {
                webSocket = new ClientWebSocket(); // ✅ ĐÃ SỬA
                Uri serverUri = new Uri(serverUrl);
                
                Debug.Log($"🔌 Connecting to {serverUrl}...");
                await webSocket.ConnectAsync(serverUri, CancellationToken.None);
                Debug.Log("✅ Connected to emotion server!");
                
                isConnected = true;
                
                _ = ReceiveEmotions();
            }
            catch (Exception ex)
            {
                Debug.LogError($"❌ Connection failed: {ex.Message}");
                Debug.LogError($"Stack trace: {ex.StackTrace}");
                isConnected = false;
            }
        }
        
        private void Update()
        {
            if (!isConnected || !webcamTexture.isPlaying) return;
            
            if (Time.time - lastSendTime >= sendInterval)
            {
                lastSendTime = Time.time;
                _ = SendFrame();
            }
        }
        
        private async Task SendFrame()
        {
            try
            {
                Color32[] pixels = webcamTexture.GetPixels32();
                
                // Convert to int array (not byte array) để JSON serialize đúng
                int[] frameInts = new int[pixels.Length * 3];
                for (int i = 0; i < pixels.Length; i++)
                {
                    frameInts[i * 3] = pixels[i].r;
                    frameInts[i * 3 + 1] = pixels[i].g;
                    frameInts[i * 3 + 2] = pixels[i].b;
                }
                
                var frameData = new
                {
                    frame = frameInts  // ✅ Gửi dạng int array
                };
                
                string json = JsonConvert.SerializeObject(frameData);
                byte[] buffer = Encoding.UTF8.GetBytes(json);
                
                await webSocket.SendAsync(
                    new ArraySegment<byte>(buffer),
                    WebSocketMessageType.Text,
                    true,
                    CancellationToken.None
                );
            }
            catch (Exception ex)
            {
                Debug.LogError($"❌ Send frame error: {ex.Message}");
            }
        }
        
        private async Task ReceiveEmotions()
        {
            byte[] buffer = new byte[1024 * 1024];
            
            while (webSocket.State == WebSocketState.Open)
            {
                try
                {
                    var result = await webSocket.ReceiveAsync(
                        new ArraySegment<byte>(buffer),
                        CancellationToken.None
                    );
                    
                    if (result.MessageType == WebSocketMessageType.Text)
                    {
                        string message = Encoding.UTF8.GetString(buffer, 0, result.Count);
                        var response = JsonConvert.DeserializeObject<EmotionResponse>(message);
                        
                        UpdatePlayerAction(response.emotion);
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogError($"❌ Receive error: {ex.Message}");
                    break;
                }
            }
        }
        
        private void UpdatePlayerAction(string emotion)
        {
            Debug.Log($"😊 Detected emotion: {emotion}");
            
            switch (emotion)
            {
                case "happy":
                    input.SetEmotionJump(true);
                    Invoke(nameof(ResetJump), 0.1f);
                    break;
                    
                case "angry":
                    input.SetEmotionMove(Vector2.up * 1.5f);
                    Invoke(nameof(ResetMove), 1f);
                    break;
                    
                case "sad":
                    input.SetEmotionMove(Vector2.down * 0.5f);
                    Invoke(nameof(ResetMove), 1f);
                    break;
                    
                case "surprise":
                    input.SetEmotionMove(Vector2.right);
                    Invoke(nameof(ResetMove), 0.5f);
                    break;
                    
                case "fear":
                    input.SetEmotionMove(Vector2.down);
                    Invoke(nameof(ResetMove), 1f);
                    break;
                    
                default:
                    input.ClearEmotion();
                    break;
            }
        }
        
        private void ResetJump()
        {
            input.SetEmotionJump(false);
        }
        
        private void ResetMove()
        {
            input.ClearEmotion();
        }
        
        private async void OnDestroy()
        {
            if (webcamTexture != null)
            {
                webcamTexture.Stop();
            }
            
            if (webSocket != null && webSocket.State == WebSocketState.Open)
            {
                await webSocket.CloseAsync(
                    WebSocketCloseStatus.NormalClosure,
                    "Client closing",
                    CancellationToken.None
                );
            }
        }
        
        [System.Serializable]
        private class EmotionResponse
        {
            public string emotion;
        }
    }
}