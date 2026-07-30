document.addEventListener('DOMContentLoaded', () => {
    const searchBtn = document.getElementById('search-btn');
    const movieInput = document.getElementById('movie-input');
    const errorMessage = document.getElementById('error-message');
    const resultsSection = document.getElementById('results-section');
    const recommendationsGrid = document.getElementById('recommendations-grid');
    const searchedMovieSpan = document.getElementById('searched-movie');
    const loadingSpinner = document.getElementById('loading-spinner');

    const handleSearch = async () => {
        const query = movieInput.value.trim();
        
        if (!query) {
            showError("Please enter a movie name.");
            return;
        }

        // Reset state
        hideError();
        resultsSection.classList.add('hidden');
        recommendationsGrid.innerHTML = '';
        loadingSpinner.classList.remove('hidden');

        try {
            const response = await fetch(`/recommend?title=${encodeURIComponent(query)}`);
            const data = await response.json();

            if (!response.ok || data.recommendations.error) {
                throw new Error(data.recommendations?.error || "Failed to fetch recommendations.");
            }

            displayRecommendations(query, data.recommendations);
        } catch (error) {
            showError(error.message || "Movie not found in our current dataset.");
        } finally {
            loadingSpinner.classList.add('hidden');
        }
    };

    const displayRecommendations = (title, movies) => {
        searchedMovieSpan.textContent = title;
        
        if (movies.length === 0) {
            recommendationsGrid.innerHTML = '<p style="text-align:center; width:100%; color:var(--text-muted)">No similar movies found.</p>';
        } else {
            movies.forEach((movie, index) => {
                const card = document.createElement('div');
                card.className = 'movie-card';
                card.style.animationDelay = `${index * 0.1}s`;
                card.style.animation = 'fadeIn 0.5s ease forwards';
                card.style.opacity = '0';
                
                const genres = movie.genre ? movie.genre.replace(/\|/g, ' • ') : 'Unknown Genre';
                
                card.innerHTML = `
                    <h3 class="movie-title">${movie.title}</h3>
                    <div class="movie-genre">${genres}</div>
                    <p class="movie-overview">${movie.overview}</p>
                `;
                
                recommendationsGrid.appendChild(card);
            });
        }
        
        resultsSection.classList.remove('hidden');
    };

    const showError = (msg) => {
        errorMessage.textContent = msg;
        errorMessage.classList.remove('hidden');
    };

    const hideError = () => {
        errorMessage.textContent = '';
        errorMessage.classList.add('hidden');
    };

    // Event listeners
    searchBtn.addEventListener('click', handleSearch);
    
    movieInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    });
});
