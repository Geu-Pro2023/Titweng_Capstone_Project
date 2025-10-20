function goToLogin() {
    // Add fade out animation
    document.querySelector('.splash-container').style.opacity = '0';
    document.querySelector('.splash-container').style.transform = 'scale(0.9)';
    
    // Redirect to login after animation
    setTimeout(() => {
        window.location.href = 'login.html';
    }, 500);
}