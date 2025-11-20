using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace StarterAssets
{
    public class StarterAssetsInputs : MonoBehaviour
    {
        [Header("Character Input Values")]
        public Vector2 move;
        public Vector2 look;
        public bool jump;
        public bool sprint;

        [Header("Movement Settings")]
        public bool analogMovement;

        [Header("Mouse Cursor Settings")]
        public bool cursorLocked = true;
        public bool cursorInputForLook = true;

        //EMOTION OVERRIDE
        private bool emotionOverride = false;
        private Vector2 emotionMove = Vector2.zero;
        private bool emotionJump = false;

#if ENABLE_INPUT_SYSTEM
        public void OnMove(InputValue value)
        {
            if (!emotionOverride)
                MoveInput(value.Get<Vector2>());
        }

        public void OnLook(InputValue value)
        {
            if (cursorInputForLook)
                LookInput(value.Get<Vector2>());
        }

        public void OnJump(InputValue value)
        {
            if (!emotionOverride)
                JumpInput(value.isPressed);
        }

        public void OnSprint(InputValue value)
        {
            if (!emotionOverride)
                SprintInput(value.isPressed);
        }
#endif

        public void MoveInput(Vector2 newMoveDirection)
        {
            move = newMoveDirection;
        }

        public void LookInput(Vector2 newLookDirection)
        {
            look = newLookDirection;
        }

        public void JumpInput(bool newJumpState)
        {
            jump = newJumpState;
        }

        public void SprintInput(bool newSprintState)
        {
            sprint = newSprintState;
        }

        //EMOTION INPUT API — EmotionClient dùng 2 hàm này
        public void SetEmotionMove(Vector2 m)
        {
            emotionOverride = true;
            emotionMove = m;
            move = emotionMove;
        }

        public void SetEmotionJump(bool j)
        {
            emotionOverride = true;
            emotionJump = j;
            jump = emotionJump;
        }

        public void ClearEmotion()
        {
            emotionOverride = false;
        }

        private void OnApplicationFocus(bool hasFocus)
        {
            SetCursorState(cursorLocked);
        }

        private void SetCursorState(bool newState)
        {
            Cursor.lockState = newState ? CursorLockMode.Locked : CursorLockMode.None;
        }
    }
}
