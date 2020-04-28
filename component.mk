COMPONENT_SRCDIRS := $(call ListAllSubDirs,$(COMPONENT_PATH)/src)
COMPONENT_INCDIRS := src
COMMON_FLAGS := -Wno-sign-compare -Wno-strict-aliasing -Wno-deprecated-declarations -Wno-nonnull -lm
ifneq ($(SMING_ARCH),Host)
	COMMON_FLAGS += -D__ANDROID__=0 
endif
COMPONENT_CFLAGS := $(COMMON_FLAGS)
COMPONENT_CXXFLAGS := $(COMMON_FLAGS) 
