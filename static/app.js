const signUpForm = document.querySelector(".sign-up-form");
const firstNameInput = document.querySelector("input[name='first_name']");
const lastNameInput = document.querySelector("input[name='last_name']");
const usernameInput = document.querySelector("input[name='username']");
const emailInput = document.querySelector("input[name='email']");
var passwordInput = document.getElementById("passwordInput");
const confirmPasswordInput = document.getElementById("confirmPasswordInput");
const email_warningMessage = document.querySelector(".email-warning");
const password_warningMessage = document.querySelector(".password-warning");
var confirmPasswordMessage = document.getElementById("confirmPasswordMessage");
const sign_in_btn = document.querySelector("#sign-in-btn");
const sign_up_btn = document.querySelector("#sign-up-btn");
const container = document.querySelector(".container");
const register_btn = document.querySelector(".btn[value='Sign Up']");
const login_btn = document.querySelector(".btn[value='lOGIN']");
var iconConfirmPassword = document.getElementById("iconConfirmPassword");
var iconPassword = document.getElementById("iconPassword");
const signUpButton = document.querySelector(".btn[value='Sign Up']");
var PasswordInput = document.querySelector("#pass");
var icon = document.getElementById("icon");
const LOGIN_usernameInput = document.querySelector("input[name='LOGIN_username']");
var password_candidate = passwordInput.value;
sign_up_btn.addEventListener('click', () => {
    container.classList.add("sign-up-mode");
});

sign_in_btn.addEventListener('click', () => {
    container.classList.remove("sign-up-mode");
});

firstNameInput.addEventListener("input", function () {
    validateLetters(this);
});

// Event listener for last name input
lastNameInput.addEventListener("input", function () {
    validateLetters(this);
});

// Function to validate input for letters only
function validateLetters(inputElement) {
    const inputValue = inputElement.value;
    const regex = /^[A-Za-z]+$/;

    if (!regex.test(inputValue)) {
        // If input contains non-letter characters, remove them
        inputElement.value = inputValue.replace(/[^A-Za-z]/g, '');
    }
}
    
emailInput.addEventListener("input", () => {
    // Check if the entered value is a valid email format
    if (!isValidEmail(emailInput.value)) {
        displayEmailWarning("Invalid email format");
    } else {
        clearEmailWarning();
    }
});

// Function to display an email warning message
function displayEmailWarning(message) {
    email_warningMessage.innerText = message;
    email_warningMessage.style.display = "block";
}

// Function to clear the email warning message
function clearEmailWarning() {
    email_warningMessage.innerText = "";
    email_warningMessage.style.display = "none";
}

// Function to check if the email is in a valid format
function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

passwordInput.addEventListener("input", () => {
    const password = passwordInput.value;

    // Check if the password meets the format requirements
    if (!isValidPassword(password)) {
        displayPasswordWarning("Password should be at least 8 characters long and contain letters, digits, and symbols");
    } else {
        clearPasswordWarning();
    }
});

// Function to display a password warning message
function displayPasswordWarning(message) {
    password_warningMessage.innerText = message;
    password_warningMessage.style.display = "block";
}

// Function to clear the password warning message
function clearPasswordWarning() {
    password_warningMessage.innerText = "";
    password_warningMessage.style.display = "none";
}

// Function to check if the password meets the format requirements
function isValidPassword(password) {
    // The regular expression checks for at least 8 characters, one letter, one digit, and one special character
    const regex = /^(?=.*[a-zA-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/;
    return regex.test(password);
}

confirmPasswordInput.addEventListener("input", function () {
    checkPasswordMatch();
});

function checkPasswordMatch() {
    var password = passwordInput.value;
    var confirmPassword = confirmPasswordInput.value;

    if (password === confirmPassword && password !== "") {
        iconConfirmPassword.style.color = "green";
        confirmPasswordMessage.textContent = "Passwords match!";
        confirmPasswordMessage.className = "success-message";
    } else {
        iconConfirmPassword.style.color = "red";
        confirmPasswordMessage.textContent = "Passwords do not match!";
        confirmPasswordMessage.className = "error-message";
    }
}
// Add input event listeners to all required fields
firstNameInput.addEventListener("input", checkAllFields);
lastNameInput.addEventListener("input", checkAllFields);
usernameInput.addEventListener("input", checkAllFields);
emailInput.addEventListener("input", checkAllFields);
passwordInput.addEventListener("input", checkAllFields);
confirmPasswordInput.addEventListener("input", checkAllFields);

// Function to check if all required fields are filled
function checkAllFields() {
    const firstName = firstNameInput.value;
    const lastName = lastNameInput.value;
    const username = usernameInput.value;
    const email = emailInput.value;
    const password = passwordInput.value;
    const confirmPassword = confirmPasswordInput.value;

    // Check if all fields are filled
    if (isEmpty(firstName) || isEmpty(lastName) || isEmpty(username) ||
        isEmpty(email) || isEmpty(password) || isEmpty(confirmPassword)) {
        signUpButton.disabled = true;
        signUpButton.style.backgroundColor = "red";
        signUpButton.style.cursor = "not-allowed";
        displayWarning("All fields are required");
    } else {
        signUpButton.disabled = false;
        signUpButton.style.backgroundColor = "#5995fd";
        signUpButton.style.cursor = "pointer";
        clearWarning();
    }
}

// Function to display a warning message
function displayWarning(message) {
    emailWarning.innerText = message;
    emailWarning.style.color = "red";
    emailWarning.style.display = "block";
}

// Function to clear the warning message
function clearWarning() {
    emailWarning.innerText = "";
    emailWarning.style.display = "none";
}

// Function to check if a field is empty
function isEmpty(value) {
    return value.trim() === "";
}


// Add input event listeners to the username and password fields
LOGIN_usernameInput.addEventListener("input", checkLoginButton);
PasswordInput.addEventListener("input", checkLoginButton);

// Function to check if both username and password are filled
function checkLoginButton() {
    const username = LOGIN_usernameInput.value.trim();
    const password = PasswordInput.value.trim();

    if (username && password) {
        login_btn.disabled = false;
        login_btn.style.backgroundColor = "#5995fd";
        login_btn.style.cursor = "pointer";
    } else {
        login_btn.disabled = true;
        login_btn.style.backgroundColor = "red";
        login_btn.style.cursor = "not-allowed";
    }
}

document.addEventListener('DOMContentLoaded', function () {
    const signUpForm = document.querySelector('.sign-up-form');
    const verifyForm = document.querySelector('.verify-form');

    signUpForm.addEventListener('submit', function(event) {


        
        signUpForm.submit();
    });
});

function submitForm() {
    // You can add validation logic here before submitting the form
    return true;
}


function submitForm() {
    // You can add validation logic here before submitting the form
    return true;
}










// Toggle password visibility


icon.onclick = function () {
    if (PasswordInput.className == 'active') {
        PasswordInput.setAttribute('type', 'text');
        icon.className = 'fa fa-eye';
        passwordInput.className = '';
    } else {
        PasswordInput.setAttribute('type', 'password');
        icon.className = 'fa fa-eye-slash';
        PasswordInput.className = 'active';
    }
};
// Toggle password visibility for Confirm Password
iconConfirmPassword.onclick = function () {
    var type = confirmPasswordInput.getAttribute('type') === 'password' ? 'text' : 'password';
    confirmPasswordInput.setAttribute('type', type);
    iconConfirmPassword.className = type === 'text' ? 'fa fa-eye' : 'fa fa-eye-slash';
};

// Toggle password visibility for Password
iconPassword.onclick = function () {
    var type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
    passwordInput.setAttribute('type', type);
    iconPassword.className = type === 'text' ? 'fa fa-eye' : 'fa fa-eye-slash';
};

function adjustContainerHeight() {
    var containerHeight = $('.forms-container').height();
    $('.container').css('min-height', containerHeight + 'px');
}

// Function to adjust social media icons position
function adjustSocialMediaPosition() {
    var socialMediaHeight = $('.social-media').outerHeight();
    $('.signin-signup').css('margin-bottom', socialMediaHeight + 'px');
}

// Call the functions when the document is ready
$(document).ready(function() {
    adjustContainerHeight();
    adjustSocialMediaPosition();


 $('#signup-form').on('submit', function() {
         adjustSocialMediaPosition();
     });

    
});

