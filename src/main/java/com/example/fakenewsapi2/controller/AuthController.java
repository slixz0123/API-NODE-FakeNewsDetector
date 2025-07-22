package com.example.fakenewsapi2.controller;


import com.example.fakenewsapi2.model.User;
import com.example.fakenewsapi2.repository.UserRepository;
import com.example.fakenewsapi2.service.JwtService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.*;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

// src/main/java/com/example/fakenewsapi2/controller/AuthController.java
@RestController
@RequestMapping("/api/auth")
public class AuthController {

    @Autowired
    private AuthenticationManager authenticationManager;

    @Autowired
    private JwtService jwtService;

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private PasswordEncoder passwordEncoder;

    @PostMapping("/login")
    public ResponseEntity<AuthResponse> login(@RequestBody LoginRequest request) {
        Authentication authentication = authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(
                        request.getUsername(),
                        request.getPassword()
                )
        );

        SecurityContextHolder.getContext().setAuthentication(authentication);
        User user = (User) authentication.getPrincipal();
        String token = jwtService.generateToken(user);
        System.out.println("Login attempt for: " + request.getUsername());
        return ResponseEntity.ok(new AuthResponse(
                token,
                user, // Enviar objeto completo
                36000 // 10 horas en segundos

        ));

    }

    @PostMapping("/signup")
    public ResponseEntity<AuthResponse> signup(@RequestBody SignupRequest request) {
        if (userRepository.existsByUsername(request.getUsername())) {
            throw new BadCredentialsException("Username already exists");
        }

        if (userRepository.existsByEmail(request.getEmail())) {
            throw new BadCredentialsException("Email already exists");
        }

        User user = new User();
        user.setName(request.getName());
        user.setUsername(request.getUsername());
        user.setEmail(request.getEmail());
        user.setPassword(passwordEncoder.encode(request.getPassword()));
        user.setAvatar(request.getAvatar());

        User savedUser = userRepository.save(user);

        String token = jwtService.generateToken(savedUser);

        return ResponseEntity.ok(new AuthResponse(
                token,
                savedUser, // Usar el objeto User directamente
                36000
        ));
    }

    // Clases de request/response
    public static class LoginRequest {
        private String username;
        private String password;
        // Getters y setters

        public String getUsername() {
            return username;
        }

        public void setUsername(String username) {
            this.username = username;
        }

        public String getPassword() {
            return password;
        }

        public void setPassword(String password) {
            this.password = password;
        }
    }

    public static class SignupRequest {
        private String name;
        private String username;
        private String email;
        private String password;
        private String avatar;
        // Getters y setters

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public String getUsername() {
            return username;
        }

        public void setUsername(String username) {
            this.username = username;
        }

        public String getEmail() {
            return email;
        }

        public void setEmail(String email) {
            this.email = email;
        }

        public String getPassword() {
            return password;
        }

        public void setPassword(String password) {
            this.password = password;
        }

        public String getAvatar() {
            return avatar;
        }

        public void setAvatar(String avatar) {
            this.avatar = avatar;
        }
    }




        // Getters



        // NUEVO USER
        public static class AuthResponse {
            private String token;
            private User user;
            private long expiresIn;

            // Constructor para login/signup
            public AuthResponse(String token, User user, long expiresIn) {
                this.token = token;
                this.user = user;
                this.expiresIn = expiresIn;
            }

            // Getters y setters
            public String getToken() { return token; }
            public void setToken(String token) { this.token = token; }

            public User getUser() { return user; }
            public void setUser(User user) { this.user = user; }

            public long getExpiresIn() { return expiresIn; }
            public void setExpiresIn(long expiresIn) { this.expiresIn = expiresIn; }
        }


    }






