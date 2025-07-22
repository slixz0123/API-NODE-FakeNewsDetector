package com.example.fakenewsapi2.config;


import org.springdoc.core.models.GroupedOpenApi;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import io.swagger.v3.oas.models.info.Info;

@Configuration
public class SwaggerConfig {

    @Bean
    public GroupedOpenApi publicApi() {
        return GroupedOpenApi.builder()
                .group("public")
                .pathsToMatch("/api/**")  // Ruta a incluir en la documentación
                .addOpenApiCustomizer(openApi -> openApi.info(new Info().title("Fake News API")
                        .description("API para la clasificación de noticias falsas")
                        .version("1.0")))
                .build();
    }
}
