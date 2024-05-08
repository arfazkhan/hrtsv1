const inputs = document.querySelectorAll('.input');
const form = document.querySelector('form');
// const [firstname, lastname, email, phone, city, country, linkedln, website] = inputs;
// const cvLabel = document.querySelector('cv-label');
// const file = document.querySelector('input[type="file"]');
const file = document.querySelector('#cv');
const checkbox = document.querySelector('input[type="checkbox"]')


for (let input of inputs) {
    input.addEventListener('focus', () => {
        let parent = input.parentElement;
        let label = parent.querySelector('label')
        label.className = "label focus"
        label.style.visibility = "visible"
    })

    input.addEventListener('blur', () => {
        let parent = input.parentElement;
        let label = parent.querySelector('label')
        label.className = "label";
        label.style.visibility = "hidden"
    })
}

form.addEventListener('submit', (e) => {
    e.preventDefault();
    validateInputs();

})

function validateInputs() {
   
    for (let input of inputs) {

        if (input.name !== "website") {
            if (input.value.trim() === "") showError(input, "This field is required");
            else {
                showSuccess(input)
            }
        }
    }

    if (file.value === "") showError(file, "No file selected");
    else {
        showSuccess(file);
        if (!checkbox.checked) showError(checkbox, "Do you agree?")
        else {
            showSuccess(checkbox);
            form.reset();
            window.open("https://codepen.io/i_amsuperfly/pen/MWrEjar", '_self');
        }
    }


}

function showError(input, message) {
    const formControl = input.parentElement;
    const errorMessage = formControl.querySelector('small');
    errorMessage.innerText = message;
    formControl.className = "form-control text error"
}

function showSuccess(input) {
    const formControl = input.parentElement;
    formControl.className = "form-control text"
}

