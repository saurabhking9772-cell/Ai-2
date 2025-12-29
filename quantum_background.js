// Quantum Background Animation
document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('quantum-canvas');
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    // Quantum particles
    const particles = [];
    const particleCount = 200;
    
    class QuantumParticle {
        constructor() {
            this.reset();
            this.entanglement = null;
            this.quantumState = Math.random() * Math.PI * 2;
        }
        
        reset() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.vx = (Math.random() - 0.5) * 2;
            this.vy = (Math.random() - 0.5) * 2;
            this.radius = Math.random() * 3 + 1;
            this.color = this.getQuantumColor();
            this.life = 1;
            this.decay = Math.random() * 0.005 + 0.002;
            this.phase = Math.random() * Math.PI * 2;
            this.frequency = Math.random() * 0.05 + 0.02;
        }
        
        getQuantumColor() {
            // Quantum-inspired colors
            const phases = [
                `hsl(${Math.random() * 60 + 180}, 100%, 70%)`, // Blue
                `hsl(${Math.random() * 60 + 300}, 100%, 70%)`, // Purple
                `hsl(${Math.random() * 60 + 60}, 100%, 70%)`   // Cyan
            ];
            return phases[Math.floor(Math.random() * phases.length)];
        }
        
        update() {
            // Quantum wave function simulation
            this.quantumState += this.frequency;
            this.phase += 0.05;
            
            // Update position with quantum uncertainty
            this.x += this.vx + Math.sin(this.quantumState) * 0.5;
            this.y += this.vy + Math.cos(this.quantumState) * 0.5;
            
            // Boundary check with quantum tunneling
            if (this.x < -this.radius) this.x = canvas.width + this.radius;
            if (this.x > canvas.width + this.radius) this.x = -this.radius;
            if (this.y < -this.radius) this.y = canvas.height + this.radius;
            if (this.y > canvas.height + this.radius) this.y = -this.radius;
            
            // Life decay
            this.life -= this.decay;
            if (this.life <= 0) this.reset();
            
            // Quantum entanglement effect
            if (this.entanglement) {
                const dx = this.entanglement.x - this.x;
                const dy = this.entanglement.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < 100) {
                    // Entangled particles move together
                    this.vx += dx * 0.001;
                    this.vy += dy * 0.001;
                }
            }
        }
        
        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            
            // Quantum glow effect
            const gradient = ctx.createRadialGradient(
                this.x, this.y, 0,
                this.x, this.y, this.radius * 3
            );
            
            gradient.addColorStop(0, this.color);
            gradient.addColorStop(0.5, this.color.replace('70%)', '50%)'));
            gradient.addColorStop(1, this.color.replace('70%)', '0%)'));
            
            ctx.fillStyle = gradient;
            ctx.globalAlpha = this.life * 0.7;
            ctx.fill();
            
            // Quantum probability wave
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius * (1 + Math.sin(this.phase) * 0.5), 0, Math.PI * 2);
            ctx.strokeStyle = this.color;
            ctx.lineWidth = 0.5;
            ctx.globalAlpha = this.life * 0.3;
            ctx.stroke();
        }
    }
    
    // Create particles
    for (let i = 0; i < particleCount; i++) {
        particles.push(new QuantumParticle());
        
        // Create entangled pairs
        if (i % 2 === 0 && i < particleCount - 1) {
            particles[i].entanglement = particles[i + 1];
            particles[i + 1].entanglement = particles[i];
        }
    }
    
    // Connection lines
    function drawConnections() {
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const p1 = particles[i];
                const p2 = particles[j];
                
                const dx = p1.x - p2.x;
                const dy = p1.y - p2.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                // Draw connection based on distance and quantum state
                if (distance < 150) {
                    const opacity = (1 - distance / 150) * 0.2;
                    
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    
                    // Quantum interference pattern
                    const phaseDiff = Math.sin(p1.phase - p2.phase);
                    const color = phaseDiff > 0 ? 
                        `rgba(0, 255, 255, ${opacity})` : 
                        `rgba(255, 0, 255, ${opacity})`;
                    
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 0.5 * (1 + phaseDiff * 0.5);
                    ctx.stroke();
                }
            }
        }
    }
    
    // Neural network overlay
    function drawNeuralNetwork() {
        const nodes = 20;
        const nodePositions = [];
        
        // Create neural nodes
        for (let i = 0; i < nodes; i++) {
            nodePositions.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                activation: Math.random()
            });
        }
        
        // Draw connections
        ctx.globalAlpha = 0.1;
        for (let i = 0; i < nodes; i++) {
            for (let j = 0; j < nodes; j++) {
                if (Math.random() > 0.7) {
                    ctx.beginPath();
                    ctx.moveTo(nodePositions[i].x, nodePositions[i].y);
                    ctx.lineTo(nodePositions[j].x, nodePositions[j].y);
                    ctx.strokeStyle = `hsl(${Math.random() * 360}, 100%, 70%)`;
                    ctx.lineWidth = Math.random() * 2;
                    ctx.stroke();
                }
            }
        }
        
        // Draw nodes
        ctx.globalAlpha = 0.3;
        nodePositions.forEach(node => {
            ctx.beginPath();
            ctx.arc(node.x, node.y, 2 + node.activation * 3, 0, Math.PI * 2);
            ctx.fillStyle = `hsl(${node.activation * 360}, 100%, 70%)`;
            ctx.fill();
        });
    }
    
    // Animation loop
    function animate() {
        // Clear with fade effect
        ctx.fillStyle = 'rgba(10, 10, 26, 0.1)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw neural network (occasionally)
        if (Math.random() > 0.95) {
            drawNeuralNetwork();
        }
        
        // Update and draw particles
        particles.forEach(particle => {
            particle.update();
            particle.draw();
        });
        
        // Draw quantum connections
        drawConnections();
        
        // Draw quantum field lines
        drawQuantumField();
        
        requestAnimationFrame(animate);
    }
    
    function drawQuantumField() {
        const gridSize = 50;
        ctx.globalAlpha = 0.05;
        ctx.strokeStyle = '#00ffff';
        ctx.lineWidth = 0.5;
        
        for (let x = 0; x < canvas.width; x += gridSize) {
            for (let y = 0; y < canvas.height; y += gridSize) {
                // Calculate field strength based on nearby particles
                let fieldX = 0;
                let fieldY = 0;
                
                particles.forEach(p => {
                    const dx = p.x - x;
                    const dy = p.y - y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    
                    if (distance < 100) {
                        const strength = (100 - distance) / 100;
                        fieldX += (dx / distance) * strength;
                        fieldY += (dy / distance) * strength;
                    }
                });
                
                // Draw field line
                const length = Math.sqrt(fieldX * fieldX + fieldY * fieldY);
                if (length > 0.5) {
                    ctx.beginPath();
                    ctx.moveTo(x, y);
                    ctx.lineTo(x + fieldX * 10, y + fieldY * 10);
                    ctx.stroke();
                    
                    // Draw arrow head
                    const angle = Math.atan2(fieldY, fieldX);
                    ctx.beginPath();
                    ctx.moveTo(x + fieldX * 10, y + fieldY * 10);
                    ctx.lineTo(
                        x + fieldX * 10 - Math.cos(angle - Math.PI/6) * 5,
                        y + fieldY * 10 - Math.sin(angle - Math.PI/6) * 5
                    );
                    ctx.lineTo(
                        x + fieldX * 10 - Math.cos(angle + Math.PI/6) * 5,
                        y + fieldY * 10 - Math.sin(angle + Math.PI/6) * 5
                    );
                    ctx.closePath();
                    ctx.fillStyle = '#00ffff';
                    ctx.fill();
                }
            }
        }
    }
    
    // Start animation
    animate();
    
    // Add interactive effects
    canvas.addEventListener('mousemove', function(e) {
        const mouseX = e.clientX;
        const mouseY = e.clientY;
        
        // Create quantum disturbance
        particles.forEach(particle => {
            const dx = particle.x - mouseX;
            const dy = particle.y - mouseY;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < 100) {
                const force = (100 - distance) / 100;
                particle.vx += (dx / distance) * force * 0.5;
                particle.vy += (dy / distance) * force * 0.5;
            }
        });
    });
});
